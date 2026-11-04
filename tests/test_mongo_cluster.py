import pytest
from unittest.mock import patch, MagicMock
from modules.mongo_cluster import MongoClusterManager

def test_mongo_cluster_offline_fallback():
    with patch("pymongo.MongoClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.admin.command.side_effect = Exception("Connection refused")
        mock_client.return_value = mock_instance
        
        cluster = MongoClusterManager(uri="mongodb://invalid:27017/", timeout_ms=50)
        assert cluster.is_available() is False
        success = cluster.save_document("ai_scientist_inputs", "inputs", {"input_id": "test"}, id_key="input_id")
        assert success is False

def test_mongo_cluster_save_and_query_mocked():
    with patch("pymongo.MongoClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.admin.command.return_value = {"ok": 1}
        mock_db = MagicMock()
        mock_col = MagicMock()
        mock_instance.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_col
        mock_col.update_one.return_value = MagicMock(acknowledged=True)
        mock_col.find.return_value = [{"input_id": "INP-1", "source": "test.csv"}]
        mock_client.return_value = mock_instance
        
        cluster = MongoClusterManager(uri="mongodb://localhost:27017/", timeout_ms=100)
        assert cluster.is_available() is True
        
        saved = cluster.save_document(
            db_name="ai_scientist_inputs",
            collection_name="inputs",
            doc={"input_id": "INP-1", "source": "test.csv"},
            id_key="input_id"
        )
        assert saved is True
        mock_col.update_one.assert_called_once()
