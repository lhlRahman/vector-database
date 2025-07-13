# Vector Database REST API Documentation

## Overview

The Vector Database API provides a RESTful interface for storing and searching high-dimensional vectors. It supports both exact (KD-tree) and approximate (LSH) nearest neighbor search, with persistent storage and metadata support.

## Quick Start

### 1. Start the server
```bash
make run-server
# Or with custom options:
./build/vector_db_server --port 8080 --dimensions 128 --file my_vectors.db
```

### 2. Test with client
```bash
# C++ client
make run-client

# Python client
pip install requests numpy
python examples/api_client_demo.py
```

## API Endpoints

### Health & Status

#### GET /health
Check server health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "Vector Database API",
  "version": "1.0.0"
}
```

#### GET /info
Get database information.

**Response:**
```json
{
  "dimensions": 128,
  "use_approximate": false,
  "vector_count": 100
}
```

### Vector Operations

#### POST /vectors
Insert a single vector.

**Request Body:**
```json
{
  "key": "vector_123",
  "vector": [0.1, 0.2, 0.3, ...],
  "metadata": "Optional metadata string"
}
```

**Response:**
```json
{
  "status": "success",
  "key": "vector_123",
  "dimensions": 128
}
```

#### POST /vectors/batch
Batch insert multiple vectors.

**Request Body:**
```json
{
  "vectors": [
    {
      "key": "vec1",
      "vector": [0.1, 0.2, ...]
    },
    {
      "key": "vec2",
      "vector": [0.3, 0.4, ...]
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "count": 2
}
```

#### GET /vectors
List all vectors with pagination.

**Query Parameters:**
- `page` (default: 1)
- `per_page` (default: 100, max: 1000)

**Response:**
```json
{
  "vectors": [
    {
      "key": "vec1",
      "vector": [0.1, 0.2, ...],
      "metadata": "optional"
    }
  ],
  "page": 1,
  "per_page": 100,
  "total": 250,
  "total_pages": 3
}
```

#### GET /vectors/{key}
Get a specific vector by key.

**Response:**
```json
{
  "key": "vector_123",
  "vector": [0.1, 0.2, 0.3, ...],
  "metadata": "Optional metadata"
}
```

### Search Operations

#### POST /search
Search for similar vectors.

**Request Body:**
```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "k": 5,
  "with_metadata": true
}
```

**Response:**
```json
{
  "results": [
    {
      "key": "vec1",
      "distance": 0.1234,
      "metadata": "optional"
    },
    {
      "key": "vec2",
      "distance": 0.2345
    }
  ],
  "count": 2
}
```

### Configuration

#### PUT /config/approximate
Toggle between exact and approximate search.

**Request Body:**
```json
{
  "enabled": true
}
```

**Response:**
```json
{
  "status": "success",
  "approximate_search": true
}
```

### Persistence

#### POST /save
Save database to disk.

**Response:**
```json
{
  "status": "success",
  "file": "api_vectors.db"
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200 OK` - Success
- `400 Bad Request` - Invalid input
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

Error responses follow this format:
```json
{
  "error": "Error message description"
}
```

## Command Line Options

```bash
./build/vector_db_server [options]
  -d, --dimensions <n>  Vector dimensions (default: 128)
  -p, --port <port>     Server port (default: 8080)
  -f, --file <path>     Database file path (default: api_vectors.db)
  -h, --help           Show help message
```

## Client Libraries

### C++ Client
```cpp
#include "../cpp-httplib/httplib.h"
#include "../json.hpp"

httplib::Client client("localhost", 8080);
json request = {
    {"key", "my_vector"},
    {"vector", {0.1, 0.2, 0.3}}
};
auto res = client.Post("/vectors", request.dump(), "application/json");
```

### Python Client
```python
from examples.api_client_demo import VectorDBClient

client = VectorDBClient(host="localhost", port=8080)
client.insert_vector("my_vector", [0.1, 0.2, 0.3], "metadata")
results = client.search([0.1, 0.2, 0.3], k=5)
```

### cURL Examples
```bash
# Insert vector
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{"key":"vec1","vector":[0.1,0.2,0.3]}'

# Search
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"vector":[0.1,0.2,0.3],"k":5}'

# Get info
curl http://localhost:8080/info
```

## Performance Considerations

1. **Batch Operations**: Use `/vectors/batch` for inserting multiple vectors
2. **Approximate Search**: Enable LSH for faster searches on large datasets
3. **Pagination**: Use pagination when listing vectors to avoid large responses
4. **Persistence**: Call `/save` periodically or rely on auto-save after insertions

## Security Notes

- The API currently runs without authentication
- Consider adding authentication/authorization for production use
- Use HTTPS in production environments
- Validate vector dimensions on client side to avoid errors

## Example Use Cases

### 1. Semantic Search
Store document embeddings with metadata containing the original text:
```json
{
  "key": "doc_123",
  "vector": [/* BERT embeddings */],
  "metadata": "Original document text..."
}
```

### 2. Image Similarity
Store image feature vectors:
```json
{
  "key": "img_456",
  "vector": [/* ResNet features */],
  "metadata": "image_url.jpg"
}
```

### 3. Recommendation Systems
Store user/item embeddings:
```json
{
  "key": "user_789",
  "vector": [/* User embedding */],
  "metadata": "{\"user_id\": 789, \"preferences\": [\"sci-fi\", \"action\"]}"
}
```