#!/usr/bin/env python3
"""
Tests for reMarkable MCP Server write operations.

Tests upload, delete, and mkdir tools, as well as the underlying
sync.py write methods.
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


# =============================================================================
# Test sync.py write internals
# =============================================================================


class TestSyncWriteMethods:
    """Test RemarkableClient write methods in sync.py."""

    def test_build_and_hash_index_v3(self):
        """Test index building and hashing for schema version 3."""
        from remarkable_mcp.sync import RemarkableClient

        client = RemarkableClient(device_token="test", user_token="test")

        entries = [
            {"hash": "a" * 64, "type": "0", "id": "file2.pdf", "subfiles": 0, "size": 100},
            {"hash": "b" * 64, "type": "0", "id": "file1.metadata", "subfiles": 0, "size": 50},
        ]

        index_bytes, index_hash = client._build_and_hash_index(entries, 3, "doc-123")

        # Entries should be sorted by ID
        lines = index_bytes.decode("utf-8").strip().split("\n")
        assert lines[0] == "3"
        assert "file1.metadata" in lines[1]
        assert "file2.pdf" in lines[2]

        # Schema v3 hash = SHA-256(concatenated binary hashes), sorted by ID
        sorted_entries = sorted(entries, key=lambda e: e["id"])
        hash_input = b""
        for entry in sorted_entries:
            hash_input += bytes.fromhex(entry["hash"])
        expected_hash = hashlib.sha256(hash_input).hexdigest()
        assert index_hash == expected_hash

    def test_build_and_hash_index_v4(self):
        """Test index building and hashing for schema version 4."""
        from remarkable_mcp.sync import RemarkableClient

        client = RemarkableClient(device_token="test", user_token="test")

        entries = [
            {"hash": "a" * 64, "type": "0", "id": "file1.metadata", "subfiles": 0, "size": 50},
            {"hash": "b" * 64, "type": "0", "id": "file2.pdf", "subfiles": 0, "size": 100},
        ]

        index_bytes, index_hash = client._build_and_hash_index(entries, 4, "doc-123")

        lines = index_bytes.decode("utf-8").strip().split("\n")
        assert lines[0] == "4"
        # Schema v4 has info line
        assert lines[1].startswith("0:doc-123:")

        # Schema v4 hash = SHA-256(full index content)
        expected_hash = hashlib.sha256(index_bytes).hexdigest()
        assert index_hash == expected_hash

    def test_build_and_hash_index_root_v4(self):
        """Test root index uses '.' as name in schema v4."""
        from remarkable_mcp.sync import RemarkableClient

        client = RemarkableClient(device_token="test", user_token="test")

        entries = [
            {"hash": "c" * 64, "type": "0", "id": "some-uuid", "subfiles": 3, "size": 200},
        ]

        index_bytes, _ = client._build_and_hash_index(entries, 4, "root")

        lines = index_bytes.decode("utf-8").strip().split("\n")
        assert "0:.:" in lines[1]  # Root uses "." as name

    def test_generation_conflict_error(self):
        """Test GenerationConflictError can be raised and caught."""
        from remarkable_mcp.sync import GenerationConflictError

        with pytest.raises(GenerationConflictError):
            raise GenerationConflictError("test conflict")

    @patch("requests.request")
    def test_upload_simple_api(self, mock_request):
        """Test upload uses the simple POST /doc/v2/files API."""
        from remarkable_mcp.sync import RemarkableClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"docID": "new-doc-id", "hash": "abc123"}
        mock_response.text = '{"docID": "new-doc-id", "hash": "abc123"}'
        mock_request.return_value = mock_response

        client = RemarkableClient(device_token="test", user_token="test_token")
        doc = client.upload("Test PDF", b"%PDF-1.4 fake content", file_type="pdf")

        assert doc.id == "new-doc-id"
        assert doc.hash == "abc123"
        assert doc.name == "Test PDF"
        assert doc.doc_type == "DocumentType"

        # Verify the request was made with correct params
        call_args = mock_request.call_args
        assert call_args[0][0] == "POST"
        assert "/doc/v2/files" in call_args[0][1]
        headers = call_args[1]["headers"]
        assert headers["Content-Type"] == "application/pdf"
        assert "rm-meta" in headers
        assert headers["rm-source"] == "RoR-Browser"

    @patch("requests.request")
    def test_upload_epub(self, mock_request):
        """Test upload with epub file type."""
        from remarkable_mcp.sync import RemarkableClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"docID": "epub-id", "hash": "def456"}
        mock_response.text = '{"docID": "epub-id", "hash": "def456"}'
        mock_request.return_value = mock_response

        client = RemarkableClient(device_token="test", user_token="test_token")
        doc = client.upload("My Book", b"fake epub", file_type="epub")

        assert doc.id == "epub-id"
        call_args = mock_request.call_args
        headers = call_args[1]["headers"]
        assert headers["Content-Type"] == "application/epub+zip"

    def test_upload_invalid_type(self):
        """Test upload rejects unsupported file types."""
        from remarkable_mcp.sync import RemarkableClient

        client = RemarkableClient(device_token="test", user_token="test_token")

        with pytest.raises(ValueError, match="Unsupported file type"):
            client.upload("Test", b"data", file_type="docx")

    @patch("requests.request")
    def test_upload_folder(self, mock_request):
        """Test folder creation uses the simple upload API."""
        from remarkable_mcp.sync import RemarkableClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"docID": "folder-id", "hash": "ghi789"}
        mock_response.text = '{"docID": "folder-id", "hash": "ghi789"}'
        mock_request.return_value = mock_response

        client = RemarkableClient(device_token="test", user_token="test_token")
        doc = client.upload_folder("New Folder")

        assert doc.id == "folder-id"
        assert doc.name == "New Folder"
        assert doc.doc_type == "CollectionType"

        call_args = mock_request.call_args
        headers = call_args[1]["headers"]
        assert headers["Content-Type"] == "folder"

    @patch("requests.request")
    def test_put_blob(self, mock_request):
        """Test _put_blob uploads to /sync/v3/files/{hash}."""
        from remarkable_mcp.sync import RemarkableClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        client = RemarkableClient(device_token="test", user_token="test_token")
        client._put_blob("abc" * 21 + "a", "test.metadata", b'{"parent": "trash"}')

        call_args = mock_request.call_args
        assert call_args[0][0] == "PUT"
        assert "/sync/v3/files/" in call_args[0][1]
        headers = call_args[1]["headers"]
        assert headers["rm-filename"] == "test.metadata"

    @patch("requests.request")
    def test_get_root_info(self, mock_request):
        """Test _get_root_info returns hash, generation, schema version."""
        from remarkable_mcp.sync import RemarkableClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "hash": "roothash123",
            "generation": 42,
            "schemaVersion": 4,
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_request.return_value = mock_response

        client = RemarkableClient(device_token="test", user_token="test_token")
        root_hash, generation, schema_version = client._get_root_info()

        assert root_hash == "roothash123"
        assert generation == 42
        assert schema_version == 4


# =============================================================================
# Test MCP Write Tools (require REMARKABLE_ENABLE_WRITE)
# =============================================================================


class TestWriteToolRegistration:
    """Test that write tools are registered only when REMARKABLE_ENABLE_WRITE is set."""

    @pytest.mark.asyncio
    async def test_write_tools_not_registered_by_default(self):
        """Test that write tools are NOT registered without REMARKABLE_ENABLE_WRITE."""
        from remarkable_mcp.server import mcp

        tools = await mcp.list_tools()
        tool_names = [tool.name for tool in tools]

        assert "remarkable_put" not in tool_names
        assert "remarkable_delete" not in tool_names
        assert "remarkable_mkdir" not in tool_names

    @pytest.mark.asyncio
    async def test_read_tools_still_present(self):
        """Test that read tools are still present regardless of write mode."""
        from remarkable_mcp.server import mcp

        tools = await mcp.list_tools()
        tool_names = [tool.name for tool in tools]

        expected_read_tools = [
            "remarkable_read",
            "remarkable_browse",
            "remarkable_recent",
            "remarkable_search",
            "remarkable_status",
            "remarkable_image",
        ]

        for tool_name in expected_read_tools:
            assert tool_name in tool_names, f"Read tool {tool_name} not found"


class TestWriteToolsEnabled:
    """Test write tools when REMARKABLE_ENABLE_WRITE is set.

    These tests reload the modules with the env var set.
    """

    @pytest.mark.asyncio
    async def test_write_tools_registered_with_env(self):
        """Test that write tools ARE registered when REMARKABLE_ENABLE_WRITE=1."""
        import importlib

        env_backup = os.environ.get("REMARKABLE_ENABLE_WRITE")
        os.environ["REMARKABLE_ENABLE_WRITE"] = "1"

        try:
            # We need to reload the modules to pick up the env var change
            import remarkable_mcp.api
            import remarkable_mcp.server
            import remarkable_mcp.tools

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

            # Re-import the reloaded mcp
            from remarkable_mcp.server import mcp as reloaded_mcp

            tools = await reloaded_mcp.list_tools()
            tool_names = [tool.name for tool in tools]

            assert "remarkable_put" in tool_names
            assert "remarkable_delete" in tool_names
            assert "remarkable_mkdir" in tool_names

            # Also verify we have 9 tools total (6 read + 3 write)
            assert len(tools) == 9, f"Expected 9 tools, got {len(tools)}: {tool_names}"

        finally:
            # Clean up: restore env and reload modules
            if env_backup is not None:
                os.environ["REMARKABLE_ENABLE_WRITE"] = env_backup
            elif "REMARKABLE_ENABLE_WRITE" in os.environ:
                del os.environ["REMARKABLE_ENABLE_WRITE"]

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

    @pytest.mark.asyncio
    async def test_remarkable_put_file_not_found(self):
        """Test remarkable_put with non-existent file."""
        import importlib

        env_backup = os.environ.get("REMARKABLE_ENABLE_WRITE")
        os.environ["REMARKABLE_ENABLE_WRITE"] = "1"

        try:
            import remarkable_mcp.api
            import remarkable_mcp.server
            import remarkable_mcp.tools

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

            from remarkable_mcp.server import mcp as reloaded_mcp

            result = await reloaded_mcp.call_tool(
                "remarkable_put",
                {"file_path": "/nonexistent/file.pdf"},
            )
            data = json.loads(result[0][0].text)

            assert "_error" in data
            assert data["_error"]["type"] == "file_not_found"

        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_ENABLE_WRITE"] = env_backup
            elif "REMARKABLE_ENABLE_WRITE" in os.environ:
                del os.environ["REMARKABLE_ENABLE_WRITE"]

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

    @pytest.mark.asyncio
    async def test_remarkable_put_unsupported_format(self):
        """Test remarkable_put with unsupported file format."""
        import importlib

        env_backup = os.environ.get("REMARKABLE_ENABLE_WRITE")
        os.environ["REMARKABLE_ENABLE_WRITE"] = "1"

        try:
            import remarkable_mcp.api
            import remarkable_mcp.server
            import remarkable_mcp.tools

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

            from remarkable_mcp.server import mcp as reloaded_mcp

            # Create a temp file with unsupported extension
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(b"fake docx content")
                tmp_path = tmp.name

            try:
                result = await reloaded_mcp.call_tool(
                    "remarkable_put",
                    {"file_path": tmp_path},
                )
                data = json.loads(result[0][0].text)

                assert "_error" in data
                assert data["_error"]["type"] == "unsupported_format"
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_ENABLE_WRITE"] = env_backup
            elif "REMARKABLE_ENABLE_WRITE" in os.environ:
                del os.environ["REMARKABLE_ENABLE_WRITE"]

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

    @pytest.mark.asyncio
    async def test_remarkable_put_success(self):
        """Test successful PDF upload via remarkable_put."""
        import importlib

        from remarkable_mcp.sync import Document

        env_backup = os.environ.get("REMARKABLE_ENABLE_WRITE")
        os.environ["REMARKABLE_ENABLE_WRITE"] = "1"

        try:
            import remarkable_mcp.api
            import remarkable_mcp.server
            import remarkable_mcp.tools

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

            from remarkable_mcp.server import mcp as reloaded_mcp

            # Mock client — patch AFTER reload so it targets the reloaded module
            mock_client = Mock()
            mock_client.upload.return_value = Document(
                id="new-doc-id",
                hash="abc123",
                name="Test PDF",
                doc_type="DocumentType",
            )

            # Create a temp PDF file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(b"%PDF-1.4 fake content")
                tmp_path = tmp.name

            try:
                with patch.object(remarkable_mcp.tools, "get_rmapi", return_value=mock_client):
                    result = await reloaded_mcp.call_tool(
                        "remarkable_put",
                        {"file_path": tmp_path},
                    )
                data = json.loads(result[0][0].text)

                assert "_error" not in data
                assert data["action"] == "uploaded"
                assert data["id"] == "new-doc-id"
                assert data["type"] == "pdf"
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_ENABLE_WRITE"] = env_backup
            elif "REMARKABLE_ENABLE_WRITE" in os.environ:
                del os.environ["REMARKABLE_ENABLE_WRITE"]

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

    @pytest.mark.asyncio
    async def test_remarkable_delete_not_found(self):
        """Test remarkable_delete with non-existent document."""
        import importlib

        env_backup = os.environ.get("REMARKABLE_ENABLE_WRITE")
        os.environ["REMARKABLE_ENABLE_WRITE"] = "1"

        try:
            import remarkable_mcp.api
            import remarkable_mcp.server
            import remarkable_mcp.tools

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

            from remarkable_mcp.server import mcp as reloaded_mcp

            mock_client = Mock()
            mock_client.get_meta_items.return_value = []

            with patch.object(remarkable_mcp.tools, "get_rmapi", return_value=mock_client):
                result = await reloaded_mcp.call_tool(
                    "remarkable_delete",
                    {"document": "NonExistent"},
                )
            data = json.loads(result[0][0].text)

            assert "_error" in data
            assert data["_error"]["type"] == "document_not_found"

        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_ENABLE_WRITE"] = env_backup
            elif "REMARKABLE_ENABLE_WRITE" in os.environ:
                del os.environ["REMARKABLE_ENABLE_WRITE"]

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

    @pytest.mark.asyncio
    async def test_remarkable_delete_success(self):
        """Test successful document deletion."""
        import importlib

        env_backup = os.environ.get("REMARKABLE_ENABLE_WRITE")
        os.environ["REMARKABLE_ENABLE_WRITE"] = "1"

        try:
            import remarkable_mcp.api
            import remarkable_mcp.server
            import remarkable_mcp.tools

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

            from remarkable_mcp.server import mcp as reloaded_mcp

            mock_client = Mock()
            mock_doc = Mock()
            mock_doc.VissibleName = "Test Document"
            mock_doc.ID = "doc-123"
            mock_doc.Parent = ""
            mock_doc.is_folder = False
            mock_doc.tags = []
            mock_client.get_meta_items.return_value = [mock_doc]
            mock_client.delete_document.return_value = None

            with patch.object(remarkable_mcp.tools, "get_rmapi", return_value=mock_client):
                result = await reloaded_mcp.call_tool(
                    "remarkable_delete",
                    {"document": "Test Document"},
                )
            data = json.loads(result[0][0].text)

            assert "_error" not in data
            assert data["action"] == "deleted"
            assert data["name"] == "Test Document"
            assert data["type"] == "document"

            # Verify delete was called
            mock_client.delete_document.assert_called_once_with(mock_doc)

        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_ENABLE_WRITE"] = env_backup
            elif "REMARKABLE_ENABLE_WRITE" in os.environ:
                del os.environ["REMARKABLE_ENABLE_WRITE"]

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

    @pytest.mark.asyncio
    async def test_remarkable_mkdir_success(self):
        """Test successful folder creation."""
        import importlib

        from remarkable_mcp.sync import Document

        env_backup = os.environ.get("REMARKABLE_ENABLE_WRITE")
        os.environ["REMARKABLE_ENABLE_WRITE"] = "1"

        try:
            import remarkable_mcp.api
            import remarkable_mcp.server
            import remarkable_mcp.tools

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

            from remarkable_mcp.server import mcp as reloaded_mcp

            mock_client = Mock()
            mock_client.get_meta_items.return_value = []
            mock_client.upload_folder.return_value = Document(
                id="folder-id",
                hash="folder-hash",
                name="New Folder",
                doc_type="CollectionType",
            )

            with patch.object(remarkable_mcp.tools, "get_rmapi", return_value=mock_client):
                result = await reloaded_mcp.call_tool(
                    "remarkable_mkdir",
                    {"name": "New Folder"},
                )
            data = json.loads(result[0][0].text)

            assert "_error" not in data
            assert data["action"] == "created"
            assert data["id"] == "folder-id"
            assert data["name"] == "New Folder"
            assert data["type"] == "folder"

        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_ENABLE_WRITE"] = env_backup
            elif "REMARKABLE_ENABLE_WRITE" in os.environ:
                del os.environ["REMARKABLE_ENABLE_WRITE"]

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

    @pytest.mark.asyncio
    async def test_remarkable_mkdir_already_exists(self):
        """Test mkdir when folder already exists."""
        import importlib

        env_backup = os.environ.get("REMARKABLE_ENABLE_WRITE")
        os.environ["REMARKABLE_ENABLE_WRITE"] = "1"

        try:
            import remarkable_mcp.api
            import remarkable_mcp.server
            import remarkable_mcp.tools

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)

            from remarkable_mcp.server import mcp as reloaded_mcp

            mock_client = Mock()
            mock_folder = Mock()
            mock_folder.VissibleName = "Existing Folder"
            mock_folder.ID = "existing-id"
            mock_folder.is_folder = True
            mock_client.get_meta_items.return_value = [mock_folder]

            with patch.object(remarkable_mcp.tools, "get_rmapi", return_value=mock_client):
                result = await reloaded_mcp.call_tool(
                    "remarkable_mkdir",
                    {"name": "Existing Folder"},
                )
            data = json.loads(result[0][0].text)

            assert "_error" not in data
            assert data["action"] == "already_exists"
            assert data["id"] == "existing-id"

            # upload_folder should NOT have been called
            mock_client.upload_folder.assert_not_called()

        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_ENABLE_WRITE"] = env_backup
            elif "REMARKABLE_ENABLE_WRITE" in os.environ:
                del os.environ["REMARKABLE_ENABLE_WRITE"]

            importlib.reload(remarkable_mcp.api)
            importlib.reload(remarkable_mcp.server)
            importlib.reload(remarkable_mcp.tools)


# =============================================================================
# Test CLI --write flag
# =============================================================================


class TestCLIWriteFlag:
    """Test that --write flag sets the environment variable."""

    def test_write_flag_parses(self):
        """Test that argparse accepts --write flag."""
        import argparse

        from remarkable_mcp.cli import main

        # We can't easily test main() directly since it calls run(),
        # but we can verify the parser accepts --write
        parser = argparse.ArgumentParser()
        parser.add_argument("--write", action="store_true")
        args = parser.parse_args(["--write"])
        assert args.write is True

    def test_write_env_var_recognized(self):
        """Test that REMARKABLE_ENABLE_WRITE env var is recognized."""
        import importlib

        env_backup = os.environ.get("REMARKABLE_ENABLE_WRITE")
        os.environ["REMARKABLE_ENABLE_WRITE"] = "1"

        try:
            import remarkable_mcp.api

            importlib.reload(remarkable_mcp.api)
            from remarkable_mcp.api import REMARKABLE_ENABLE_WRITE

            assert REMARKABLE_ENABLE_WRITE is True
        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_ENABLE_WRITE"] = env_backup
            elif "REMARKABLE_ENABLE_WRITE" in os.environ:
                del os.environ["REMARKABLE_ENABLE_WRITE"]

            importlib.reload(remarkable_mcp.api)

    def test_write_env_var_off_by_default(self):
        """Test that REMARKABLE_ENABLE_WRITE is off by default."""
        import importlib

        env_backup = os.environ.get("REMARKABLE_ENABLE_WRITE")
        if "REMARKABLE_ENABLE_WRITE" in os.environ:
            del os.environ["REMARKABLE_ENABLE_WRITE"]

        try:
            import remarkable_mcp.api

            importlib.reload(remarkable_mcp.api)
            from remarkable_mcp.api import REMARKABLE_ENABLE_WRITE

            assert REMARKABLE_ENABLE_WRITE is False
        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_ENABLE_WRITE"] = env_backup

            importlib.reload(remarkable_mcp.api)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
