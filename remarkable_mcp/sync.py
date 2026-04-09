"""
reMarkable Cloud Sync Client

A replacement for rmapy that uses the current reMarkable sync API (v3/v4).
rmapy is abandoned and uses deprecated endpoints that return 500 errors.

Based on the protocol used by ddvk/rmapi.
"""

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# API endpoints
# Note: my.remarkable.com endpoints redirect to doesnotexist.remarkable.com
# So we use webapp-prod.cloud.remarkable.engineering for auth
AUTH_HOST = "https://webapp-prod.cloud.remarkable.engineering"
DEVICE_TOKEN_URL = f"{AUTH_HOST}/token/json/2/device/new"
USER_TOKEN_URL = f"{AUTH_HOST}/token/json/2/user/new"

SYNC_HOST = "https://internal.cloud.remarkable.com"
ROOT_URL = f"{SYNC_HOST}/sync/v4/root"
ROOT_PUT_URL = f"{SYNC_HOST}/sync/v3/root"
FILES_URL = f"{SYNC_HOST}/sync/v3/files"
UPLOAD_URL = f"{SYNC_HOST}/doc/v2/files"


class GenerationConflictError(Exception):
    """Raised when the root hash generation doesn't match (concurrent edit)."""

    pass


@dataclass
class Document:
    """Represents a document or folder in the reMarkable cloud."""

    id: str
    hash: str
    name: str
    doc_type: str  # "DocumentType" or "CollectionType"
    parent: str = ""
    deleted: bool = False
    pinned: bool = False
    last_modified: Optional[datetime] = None
    size: int = 0
    files: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @property
    def is_folder(self) -> bool:
        return self.doc_type == "CollectionType"

    @property
    def VissibleName(self) -> str:
        """Compatibility with rmapy naming."""
        return self.name

    @property
    def ID(self) -> str:
        """Compatibility with rmapy naming."""
        return self.id

    @property
    def Parent(self) -> str:
        """Compatibility with rmapy naming."""
        return self.parent

    @property
    def Type(self) -> str:
        """Compatibility with rmapy naming."""
        return self.doc_type

    @property
    def ModifiedClient(self) -> Optional[datetime]:
        """Compatibility with rmapy naming."""
        return self.last_modified


# Alias for backward compatibility with rmapy-style code
# In our sync module, both Document and Folder are the same class,
# distinguished by the is_folder property
Folder = Document


class RemarkableClient:
    """Client for reMarkable Cloud sync API."""

    def __init__(self, device_token: str = "", user_token: str = ""):
        self.device_token = device_token
        self.user_token = user_token
        self._documents: List[Document] = []
        self._documents_by_id: Dict[str, Document] = {}

    def renew_token(self) -> str:
        """Exchange device token for a fresh user token."""
        if not self.device_token:
            raise RuntimeError("No device token available")

        headers = {"Authorization": f"Bearer {self.device_token}"}

        try:
            response = requests.post(USER_TOKEN_URL, headers=headers, timeout=30)
            if response.status_code == 200 and response.text:
                self.user_token = response.text.strip()
                return self.user_token
        except requests.RequestException as e:
            raise RuntimeError(f"Network error during token renewal: {e}")

        raise RuntimeError(
            f"Failed to renew user token (HTTP {response.status_code}).\n"
            "Your device may need to be re-registered.\n"
            "Get a new code from: https://my.remarkable.com/device/desktop/connect"
        )

    def _request(
        self,
        url: str,
        method: str = "GET",
        data: Optional[bytes] = None,
        json_data: Optional[dict] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: int = 60,
    ) -> requests.Response:
        """Make an authenticated request."""
        if not self.user_token:
            self.renew_token()

        headers = {"Authorization": f"Bearer {self.user_token}"}
        if extra_headers:
            headers.update(extra_headers)

        kwargs: Dict[str, Any] = {"headers": headers, "timeout": timeout}
        if data is not None:
            kwargs["data"] = data
        if json_data is not None:
            kwargs["json"] = json_data

        response = requests.request(method, url, **kwargs)

        if response.status_code == 401:
            # Token expired, try to renew
            self.renew_token()
            headers["Authorization"] = f"Bearer {self.user_token}"
            kwargs["headers"] = headers
            response = requests.request(method, url, **kwargs)

        return response

    def _get_file(self, file_hash: str) -> bytes:
        """Download a file by its hash."""
        response = self._request(f"{FILES_URL}/{file_hash}")
        response.raise_for_status()
        return response.content

    def _parse_index(self, content: bytes) -> List[Dict[str, Any]]:
        """Parse an index file into entries."""
        lines = content.decode("utf-8").strip().split("\n")
        entries = []

        # First line is schema version
        for line in lines[1:]:
            parts = line.split(":")
            if len(parts) >= 5:
                entries.append(
                    {
                        "hash": parts[0],
                        "type": parts[1],
                        "id": parts[2],
                        "subfiles": int(parts[3]),
                        "size": int(parts[4]),
                    }
                )

        return entries

    def get_meta_items(self, limit: Optional[int] = None) -> List[Document]:
        """
        Fetch documents and folders from the cloud.

        Args:
            limit: Maximum number of documents to fetch. If None, fetches all.

        Returns a list of Document objects (compatible with rmapy Collection).
        """
        # Get root hash
        response = self._request(ROOT_URL)
        response.raise_for_status()

        # Handle empty or invalid JSON response
        if not response.text or not response.text.strip():
            raise RuntimeError(
                "Empty response from reMarkable API. Your token may have expired.\n"
                "Try re-registering: uvx remarkable-mcp --register <code>"
            )

        try:
            root_data = response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON from reMarkable API: {e}\nResponse was: {response.text[:200]}"
            )

        if "hash" not in root_data:
            raise RuntimeError(
                f"Unexpected API response format: {root_data}\nThe reMarkable API may have changed."
            )

        root_hash = root_data["hash"]

        # Get root index
        root_index = self._get_file(root_hash)
        entries = self._parse_index(root_index)

        documents = []

        for entry in entries:
            doc_id = entry["id"]
            doc_hash = entry["hash"]

            # Fetch the document's blob index
            try:
                blob_content = self._get_file(doc_hash)
                blob_entries = self._parse_index(blob_content)
            except Exception:
                continue

            # Find and fetch the metadata file
            metadata = {}
            files = []

            for blob_entry in blob_entries:
                files.append(blob_entry)
                if blob_entry["id"].endswith(".metadata"):
                    try:
                        meta_content = self._get_file(blob_entry["hash"])
                        metadata = json.loads(meta_content.decode("utf-8"))
                    except Exception:
                        pass

            # Skip deleted documents
            if metadata.get("deleted", False):
                continue

            # Parse last modified timestamp
            last_modified = None
            if "lastModified" in metadata:
                try:
                    ts = int(metadata["lastModified"]) / 1000  # Convert ms to seconds
                    last_modified = datetime.fromtimestamp(ts)
                except (ValueError, TypeError):
                    pass

            doc = Document(
                id=doc_id,
                hash=doc_hash,
                name=metadata.get("visibleName", doc_id),
                doc_type=metadata.get("type", "DocumentType"),
                parent=metadata.get("parent", ""),
                deleted=metadata.get("deleted", False),
                pinned=metadata.get("pinned", False),
                last_modified=last_modified,
                size=entry["size"],
                files=files,
                tags=metadata.get("tags", []),
            )

            documents.append(doc)

            # Stop early if we have enough
            if limit is not None and len(documents) >= limit:
                break

        self._documents = documents
        self._documents_by_id = {d.id: d for d in documents}

        return documents

    def get_doc(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        if not self._documents_by_id:
            self.get_meta_items()
        return self._documents_by_id.get(doc_id)

    def download(self, doc: Document) -> bytes:
        """Download a document's content as a zip file."""
        # The document blob contains all the files
        # We need to fetch each file and create a zip
        import io
        import zipfile

        blob_content = self._get_file(doc.hash)
        blob_entries = self._parse_index(blob_content)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for entry in blob_entries:
                file_id = entry["id"]
                file_hash = entry["hash"]

                # Download the file
                try:
                    file_content = self._get_file(file_hash)
                    zf.writestr(file_id, file_content)
                except Exception:
                    continue

        zip_buffer.seek(0)
        return zip_buffer.read()

    # =========================================================================
    # Write operations
    # =========================================================================

    def _put_blob(self, file_hash: str, filename: str, data: bytes) -> None:
        """Upload a blob by its SHA-256 hash."""
        response = self._request(
            f"{FILES_URL}/{file_hash}",
            method="PUT",
            data=data,
            extra_headers={"rm-filename": filename},
            timeout=120,
        )
        response.raise_for_status()

    def _get_root_info(self) -> Tuple[str, int, int]:
        """Get root hash, generation counter, and schema version."""
        response = self._request(ROOT_URL)
        response.raise_for_status()
        root_data = response.json()
        return (
            root_data["hash"],
            root_data["generation"],
            root_data.get("schemaVersion", 3),
        )

    def _build_and_hash_index(
        self,
        entries: List[Dict[str, Any]],
        schema_version: int,
        doc_id: str = "root",
    ) -> Tuple[bytes, str]:
        """Build index file content and compute its hash.

        Returns (index_bytes, index_hash).
        """
        sorted_entries = sorted(entries, key=lambda e: e["id"])

        lines = [f"{schema_version}\n"]
        if schema_version == 4:
            name = "." if doc_id == "root" else doc_id
            total_size = sum(e["size"] for e in sorted_entries)
            lines.append(f"0:{name}:{len(sorted_entries)}:{total_size}\n")
        for entry in sorted_entries:
            lines.append(
                f"{entry['hash']}:{entry['type']}:{entry['id']}"
                f":{entry['subfiles']}:{entry['size']}\n"
            )
        index_bytes = "".join(lines).encode("utf-8")

        if schema_version == 3:
            # Schema v3: hash of concatenated binary hashes
            hash_input = b""
            for entry in sorted_entries:
                hash_input += bytes.fromhex(entry["hash"])
            index_hash = hashlib.sha256(hash_input).hexdigest()
        else:
            # Schema v4: hash of the full index file
            index_hash = hashlib.sha256(index_bytes).hexdigest()

        return index_bytes, index_hash

    def _put_root_hash(self, root_hash: str, generation: int) -> None:
        """Update the root hash with optimistic generation lock."""
        response = self._request(
            ROOT_PUT_URL,
            method="PUT",
            json_data={
                "hash": root_hash,
                "generation": generation,
                "broadcast": True,
            },
            timeout=30,
        )
        if response.status_code == 412 or (
            response.status_code == 200
            and "precondition failed" in response.text.lower()
        ):
            raise GenerationConflictError(
                "Root hash generation conflict — another client synced concurrently"
            )
        response.raise_for_status()

    def _edit_metadata(
        self,
        doc_id: str,
        updates: Dict[str, Any],
        max_retries: int = 3,
    ) -> None:
        """Edit a document's metadata via the hash-tree sync protocol.

        Handles generation conflicts with automatic retry.
        """
        for attempt in range(max_retries):
            try:
                # 1. Get current root state
                root_hash, generation, schema_version = self._get_root_info()

                # 2. Get root entries
                root_index = self._get_file(root_hash)
                root_entries = self._parse_index(root_index)

                # 3. Find the document entry
                doc_entry_idx = None
                for i, entry in enumerate(root_entries):
                    if entry["id"] == doc_id:
                        doc_entry_idx = i
                        break
                if doc_entry_idx is None:
                    raise RuntimeError(f"Document {doc_id} not found in root index")
                doc_entry = root_entries[doc_entry_idx]

                # 4. Get document's sub-entries
                doc_blob = self._get_file(doc_entry["hash"])
                doc_entries = self._parse_index(doc_blob)

                # 5. Find .metadata sub-entry
                meta_entry_idx = None
                for i, entry in enumerate(doc_entries):
                    if entry["id"].endswith(".metadata"):
                        meta_entry_idx = i
                        break
                if meta_entry_idx is None:
                    raise RuntimeError(f"Metadata not found for document {doc_id}")
                meta_entry = doc_entries[meta_entry_idx]

                # 6. Download, parse, and update metadata
                meta_content = self._get_file(meta_entry["hash"])
                metadata = json.loads(meta_content.decode("utf-8"))
                metadata.update(updates)

                # 7. Upload new metadata blob
                new_meta_bytes = json.dumps(metadata).encode("utf-8")
                new_meta_hash = hashlib.sha256(new_meta_bytes).hexdigest()
                self._put_blob(new_meta_hash, meta_entry["id"], new_meta_bytes)

                # 8. Rebuild document index with updated metadata hash
                doc_entries[meta_entry_idx] = {
                    **meta_entry,
                    "hash": new_meta_hash,
                    "size": len(new_meta_bytes),
                }
                new_doc_index, new_doc_hash = self._build_and_hash_index(
                    doc_entries, schema_version, doc_id
                )
                self._put_blob(new_doc_hash, f"{doc_id}.docSchema", new_doc_index)

                # 9. Rebuild root index with updated document hash
                new_doc_size = sum(e["size"] for e in doc_entries)
                root_entries[doc_entry_idx] = {
                    **doc_entry,
                    "hash": new_doc_hash,
                    "size": new_doc_size,
                    "subfiles": len(doc_entries),
                }
                new_root_index, new_root_hash = self._build_and_hash_index(
                    root_entries, schema_version, "root"
                )
                self._put_blob(new_root_hash, "root.docSchema", new_root_index)

                # 10. Commit the new root hash
                self._put_root_hash(new_root_hash, generation)
                return  # Success

            except GenerationConflictError:
                if attempt < max_retries - 1:
                    continue
                raise

    def upload(self, name: str, data: bytes, file_type: str = "pdf") -> Document:
        """Upload a file using the simple upload API.

        Args:
            name: Display name on the tablet.
            data: Raw file bytes.
            file_type: "pdf" or "epub".

        Returns:
            Document object for the uploaded file.
        """
        mime_types = {
            "pdf": "application/pdf",
            "epub": "application/epub+zip",
        }
        mime = mime_types.get(file_type)
        if not mime:
            raise ValueError(f"Unsupported file type: {file_type}. Use 'pdf' or 'epub'.")

        meta = base64.b64encode(json.dumps({"file_name": name}).encode()).decode()
        response = self._request(
            UPLOAD_URL,
            method="POST",
            data=data,
            extra_headers={
                "Content-Type": mime,
                "rm-meta": meta,
                "rm-source": "RoR-Browser",
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        return Document(
            id=result.get("docID", ""),
            hash=result.get("hash", ""),
            name=name,
            doc_type="DocumentType",
        )

    def upload_folder(self, name: str) -> Document:
        """Create a folder using the simple upload API.

        Args:
            name: Folder display name on the tablet.

        Returns:
            Document object for the created folder.
        """
        meta = base64.b64encode(json.dumps({"file_name": name}).encode()).decode()
        response = self._request(
            UPLOAD_URL,
            method="POST",
            data=b"",
            extra_headers={
                "Content-Type": "folder",
                "rm-meta": meta,
                "rm-source": "RoR-Browser",
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        return Document(
            id=result.get("docID", ""),
            hash=result.get("hash", ""),
            name=name,
            doc_type="CollectionType",
        )

    def delete_document(self, doc: Document) -> None:
        """Soft-delete a document by moving it to trash.

        Args:
            doc: The document to delete.
        """
        self._edit_metadata(doc.id, {"parent": "trash"})


def register_device(one_time_code: str) -> Dict[str, str]:
    """
    Register a new device with reMarkable cloud.

    Args:
        one_time_code: Code from https://my.remarkable.com/device/desktop/connect

    Returns:
        Dict with devicetoken and usertoken keys
    """
    from uuid import uuid4

    body = {
        "code": one_time_code,
        "deviceDesc": "desktop-linux",
        "deviceID": str(uuid4()),
    }

    try:
        response = requests.post(DEVICE_TOKEN_URL, json=body, timeout=30)
        if response.status_code == 200 and response.text:
            device_token = response.text.strip()
            return {"devicetoken": device_token, "usertoken": ""}
    except requests.RequestException as e:
        raise RuntimeError(f"Network error during registration: {e}")

    raise RuntimeError(
        f"Registration failed (HTTP {response.status_code}). This usually means:\n"
        "  1. The code has expired (codes are single-use)\n"
        "  2. The code was already used\n"
        "  3. The code was typed incorrectly\n\n"
        "Get a new code from: https://my.remarkable.com/device/desktop/connect"
    )


def load_client_from_token(token_data: str) -> RemarkableClient:
    """
    Create a client from a token string.

    Args:
        token_data: Either:
            - JSON string with devicetoken and optional usertoken
            - Raw JWT device token (legacy format from rmapy)

    Returns:
        Configured RemarkableClient
    """
    token_data = token_data.strip()

    # Try to parse as JSON first
    if token_data.startswith("{"):
        try:
            data = json.loads(token_data)
            return RemarkableClient(
                device_token=data.get("devicetoken", ""),
                user_token=data.get("usertoken", ""),
            )
        except json.JSONDecodeError:
            pass

    # Treat as raw device token (legacy rmapy format - just the JWT)
    # JWT tokens start with "eyJ" (base64 encoded '{"')
    if token_data.startswith("eyJ"):
        return RemarkableClient(device_token=token_data, user_token="")

    raise ValueError(
        f"Invalid token format. Expected JSON or JWT token.\n"
        f"Token starts with: {token_data[:20]}..."
    )


def load_client_from_file(token_file: Path = Path.home() / ".rmapi") -> RemarkableClient:
    """
    Load a client from a token file.

    Args:
        token_file: Path to JSON token file (default: ~/.rmapi)

    Returns:
        Configured RemarkableClient
    """
    if not token_file.exists():
        raise RuntimeError(
            f"Token file not found: {token_file}\n"
            "Register first with: uvx remarkable-mcp --register <code>"
        )

    token_json = token_file.read_text()
    return load_client_from_token(token_json)
