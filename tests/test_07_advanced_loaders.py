import os
import shutil
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from longtrainer.loaders import DocumentLoader

@pytest.fixture
def temp_dir(tmp_path):
    """Fixture for creating a temporary directory with test files."""
    test_dir = tmp_path / "test_docs"
    test_dir.mkdir()
    
    file1 = test_dir / "doc1.txt"
    file1.write_text("Hello World from doc1")
    
    file2 = test_dir / "doc2.md"
    file2.write_text("# Doc 2\nThis is markdown.")
    
    return str(test_dir)

def test_load_directory(temp_dir):
    """Test loading a directory of text files."""
    loader = DocumentLoader()
    
    # Needs to match all files ideally, let's test specific python glob if applicable or default
    docs = loader.load_directory(temp_dir)
    
    assert len(docs) == 2
    contents = [d.page_content for d in docs]
    assert any("Hello World from doc1" in c for c in contents)
    assert any("This is markdown." in c for c in contents)

def test_load_dynamic_loader_success():
    """Test dynamic loader with a mock."""
    loader = DocumentLoader()
    
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_loader_class = MagicMock()
        
        # Mock instance of WebBaseLoader
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [Document(page_content="Mocked Web Content")]
        
        mock_loader_class.return_value = mock_loader_instance
        mock_module.WebBaseLoader = mock_loader_class
        mock_import.return_value = mock_module
        
        docs = loader.load_dynamic_loader("WebBaseLoader", web_paths=["http://example.com"])
        
        assert len(docs) == 1
        assert docs[0].page_content == "Mocked Web Content"
        mock_loader_class.assert_called_once_with(web_paths=["http://example.com"])

def test_load_dynamic_loader_error():
    """Test dynamic loader with invalid class name."""
    loader = DocumentLoader()
    docs = loader.load_dynamic_loader("InvalidLoader12345")
    assert docs == []
