import logging
from typing import Optional, List, Dict, Any
try:
    import pymongo
    from pymongo.errors import PyMongoError
    PYMONGO_AVAILABLE = True
except ImportError:
    pymongo = None
    PyMongoError = Exception
    PYMONGO_AVAILABLE = False

from .config import MONGO_URI, MONGO_TIMEOUT_MS

logger = logging.getLogger(__name__)

class MongoClusterManager:
    """
    Manages connections and multi-database persistence across local/remote MongoDB cluster.
    Provides graceful offline handling when MongoDB is not running.
    """
    def __init__(self, uri: str = MONGO_URI, timeout_ms: int = MONGO_TIMEOUT_MS):
        self.uri = uri
        self.timeout_ms = timeout_ms
        self._client: Optional[pymongo.MongoClient] = None
        self._connected: Optional[bool] = None

    def _get_client(self):
        if not PYMONGO_AVAILABLE:
            if self._connected is not False:
                logger.warning("pymongo is not installed in the active Python environment. Operating in standalone file-based mode.")
                self._connected = False
            return None
        if self._connected is False:
            return None
        if self._client is not None:
            return self._client
        try:
            client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=self.timeout_ms)
            client.admin.command("ping")
            self._client = client
            self._connected = True
            return self._client
        except Exception as e:
            logger.warning(f"MongoDB cluster unreachable at {self.uri}: {e}. Running in standalone mode.")
            self._connected = False
            self._client = None
            return None

    def is_available(self) -> bool:
        return self._get_client() is not None

    def get_collection(self, db_name: str, collection_name: str):
        client = self._get_client()
        if client is None:
            return None
        try:
            return client[db_name][collection_name]
        except Exception as e:
            logger.warning(f"Error accessing collection {db_name}.{collection_name}: {e}")
            return None

    def save_document(
        self,
        db_name: str,
        collection_name: str,
        doc: Dict[str, Any],
        id_key: str = "_id",
        run_id: Optional[str] = None
    ) -> bool:
        col = self.get_collection(db_name, collection_name)
        if col is None:
            return False
        try:
            if run_id:
                doc["run_id"] = run_id
            filter_query: Dict[str, Any] = {}
            if "run_id" in doc:
                filter_query["run_id"] = doc["run_id"]
            if id_key in doc and id_key != "_id":
                filter_query[id_key] = doc[id_key]
            elif "_id" in doc:
                filter_query["_id"] = doc["_id"]

            if filter_query:
                col.update_one(filter_query, {"$set": doc}, upsert=True)
            else:
                col.insert_one(doc)
            return True
        except PyMongoError as e:
            logger.warning(f"Failed to save document to {db_name}.{collection_name}: {e}")
            return False

    def save_documents(
        self,
        db_name: str,
        collection_name: str,
        docs: List[Dict[str, Any]],
        id_key: str = "_id",
        run_id: Optional[str] = None
    ) -> bool:
        if not docs:
            return True
        col = self.get_collection(db_name, collection_name)
        if col is None:
            return False
        try:
            for doc in docs:
                self.save_document(db_name, collection_name, doc, id_key=id_key, run_id=run_id)
            return True
        except PyMongoError as e:
            logger.warning(f"Failed to save documents to {db_name}.{collection_name}: {e}")
            return False

    def query_documents(
        self,
        db_name: str,
        collection_name: str,
        query: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        col = self.get_collection(db_name, collection_name)
        if col is None:
            return []
        try:
            q = dict(query or {})
            if run_id:
                q["run_id"] = run_id
            return list(col.find(q, {"_id": 0}))
        except PyMongoError as e:
            logger.warning(f"Failed to query {db_name}.{collection_name}: {e}")
            return []
