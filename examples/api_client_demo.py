"""
Vector Database Test Script

This script tests the vector database by:
1. Generating embeddings using Nebius/Qwen model
2. Inserting vectors into the database via API
3. Providing interactive terminal querying

Requirements:
- pip install openai requests numpy
- Set NEBIUS_API_KEY environment variable
- Vector database server running on localhost:8080
"""

import os
import json
import requests
import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional, Tuple
import time

class VectorDBTester:
    def __init__(self, 
                 vector_db_host: str = "localhost", 
                 vector_db_port: int = 8080,
                 nebius_api_key: Optional[str] = None):
        """
        Initialize the Vector DB Tester
        
        Args:
            vector_db_host: Vector database host
            vector_db_port: Vector database port
            nebius_api_key: Nebius API key (or set NEBIUS_API_KEY env var)
        """
        self.vector_db_base_url = f"http://{vector_db_host}:{vector_db_port}"
        
        # Initialize Nebius/OpenAI client
        api_key = nebius_api_key or os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise ValueError("NEBIUS_API_KEY environment variable must be set or passed as parameter")
        
        self.embedding_client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=api_key
        )
        
        # Sample data for testing
        self.sample_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Python is a powerful programming language for data science",
            "Machine learning algorithms can solve complex problems",
            "Vector databases enable efficient similarity search",
            "Natural language processing transforms text into meaningful insights",
            "Deep learning models require large amounts of training data",
            "Artificial intelligence is revolutionizing many industries",
            "Database indexing improves query performance significantly",
            "Cloud computing provides scalable infrastructure solutions",
            "Open source software drives innovation in technology",
            "Embeddings capture semantic meaning in numerical form",
            "REST APIs provide standardized communication interfaces",
            "Data visualization helps understand complex patterns",
            "Distributed systems handle large-scale applications",
            "Version control systems track code changes over time"
        ]

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text using Nebius/Qwen model"""
        try:
            response = self.embedding_client.embeddings.create(
                model="Qwen/Qwen3-Embedding-8B",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for text '{text[:50]}...': {e}")
            return None

    def check_server_health(self) -> bool:
        """Check if the vector database server is running"""
        try:
            response = requests.get(f"{self.vector_db_base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Server is healthy: {data.get('service', 'Unknown')}")
                return True
            else:
                print(f"✗ Server returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot connect to vector database server: {e}")
            return False

    def get_db_info(self) -> Optional[Dict]:
        """Get database information"""
        try:
            response = requests.get(f"{self.vector_db_base_url}/info")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting DB info: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting DB info: {e}")
            return None

    def insert_vector(self, key: str, vector: List[float], metadata: str = "") -> bool:
        """Insert a single vector into the database"""
        try:
            payload = {
                "key": key,
                "vector": vector,
                "metadata": metadata
            }
            response = requests.post(
                f"{self.vector_db_base_url}/vectors",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return True
            else:
                print(f"Insert failed for {key}: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error inserting vector {key}: {e}")
            return False

    def batch_insert_vectors(self, vectors_data: List[Dict]) -> bool:
        """Batch insert multiple vectors"""
        try:
            # API expects direct array, not wrapped in "vectors" object
            payload = vectors_data
            response = requests.post(
                f"{self.vector_db_base_url}/vectors/batch",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"✓ Batch insert successful")
                return True
            else:
                print(f"✗ Batch insert failed: {response.status_code}")
                print(f"Error response: {response.text}")
                # Try the wrapped format as fallback
                print("🔄 Trying alternative format...")
                payload_wrapped = {"vectors": vectors_data}
                response2 = requests.post(
                    f"{self.vector_db_base_url}/vectors/batch",
                    json=payload_wrapped,
                    headers={"Content-Type": "application/json"}
                )
                if response2.status_code == 200:
                    print(f"✓ Batch insert successful with wrapped format")
                    return True
                else:
                    print(f"✗ Both formats failed: {response2.status_code}")
                    print(f"Error response: {response2.text}")
                return False
                
        except Exception as e:
            print(f"Error batch inserting vectors: {e}")
            return False

    def search_vectors(self, query_vector: List[float], k: int = 5, with_metadata: bool = True) -> Optional[Dict]:
        """Search for similar vectors"""
        try:
            payload = {
                "vector": query_vector,
                "k": k,
                "with_metadata": with_metadata
            }
            response = requests.post(
                f"{self.vector_db_base_url}/search",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Search error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return None

    def populate_database(self) -> bool:
        """Populate the database with sample embeddings"""
        print("🔄 Generating embeddings and populating database...")
        
        # First, check database dimensions
        db_info = self.get_db_info()
        expected_dims = db_info.get('dimensions') if db_info else None
        print(f"Database expects {expected_dims} dimensions")
        
        vectors_data = []
        
        for i, text in enumerate(self.sample_texts):
            print(f"Processing {i+1}/{len(self.sample_texts)}: {text[:50]}...")
            
            # Generate embedding
            embedding = self.generate_embedding(text)
            if embedding is None:
                print(f"Failed to generate embedding for: {text[:50]}...")
                continue
            
            # Check dimension compatibility
            if expected_dims and len(embedding) != expected_dims:
                print(f"⚠️  Dimension mismatch: embedding has {len(embedding)} dims, database expects {expected_dims}")
                print("You may need to restart the server with the correct dimensions:")
                print(f"./build/vector_db_server --dimensions {len(embedding)}")
                return False
            
            vectors_data.append({
                "key": f"sample_{i+1}",
                "vector": embedding,
                "metadata": text
            })
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Try single insert first to test
        if vectors_data:
            print("🧪 Testing single vector insert first...")
            test_success = self.insert_vector(
                vectors_data[0]["key"], 
                vectors_data[0]["vector"], 
                vectors_data[0]["metadata"]
            )
            
            if not test_success:
                print("✗ Single vector insert failed - checking for errors")
                return False
            else:
                print("✓ Single vector insert successful")
        
        # Batch insert all vectors
        if vectors_data:
            print(f"📦 Batch inserting {len(vectors_data)} vectors...")
            success = self.batch_insert_vectors(vectors_data)
            if success:
                print(f"✓ Successfully inserted {len(vectors_data)} vectors into database")
                return True
            else:
                print("✗ Batch insert failed, trying individual inserts...")
                # Fallback to individual inserts
                success_count = 0
                for vector_data in vectors_data[1:]:  # Skip first one already inserted
                    if self.insert_vector(vector_data["key"], vector_data["vector"], vector_data.get("metadata", "")):
                        success_count += 1
                    else:
                        print(f"Failed to insert {vector_data['key']}")
                
                if success_count > 0:
                    print(f"✓ Successfully inserted {success_count + 1} vectors individually")
                    return True
                else:
                    print("✗ All individual inserts failed")
                    return False
        else:
            print("✗ No vectors to insert")
            return False

    def interactive_query(self):
        """Interactive terminal interface for querying the database"""
        print("\n" + "="*60)
        print("🔍 INTERACTIVE VECTOR SEARCH")
        print("="*60)
        print("Enter your search queries below (type 'quit' to exit)")
        print("Examples:")
        print("  - 'machine learning and AI'")
        print("  - 'programming languages'")
        print("  - 'database performance'")
        print("-" * 60)
        
        while True:
            try:
                query = input("\n🔎 Enter your search query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    print("Please enter a valid query.")
                    continue
                
                print(f"\n🔄 Generating embedding for: '{query}'")
                query_embedding = self.generate_embedding(query)
                
                if query_embedding is None:
                    print("❌ Failed to generate embedding for query")
                    continue
                
                print("🔍 Searching database...")
                results = self.search_vectors(query_embedding, k=5)
                
                if results and results.get('results'):
                    print(f"\n📊 Found {len(results['results'])} similar vectors:")
                    print("-" * 50)
                    
                    for i, result in enumerate(results['results'], 1):
                        key = result.get('key', 'Unknown')
                        distance = result.get('distance', 0)
                        similarity = 1 / (1 + distance)  # Convert distance to similarity score
                        metadata = result.get('metadata', 'No metadata')
                        
                        print(f"{i}. Key: {key}")
                        print(f"   Similarity: {similarity:.4f} (distance: {distance:.4f})")
                        print(f"   Text: {metadata}")
                        print()
                else:
                    print("❌ No results found or search failed")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during search: {e}")

    def run_test_suite(self):
        """Run the complete test suite"""
        print("🚀 VECTOR DATABASE TEST SUITE")
        print("=" * 50)
        
        # 1. Check server health
        print("\n1️⃣ Checking server health...")
        if not self.check_server_health():
            print("❌ Cannot proceed - server is not accessible")
            return False
        
        # 2. Get database info
        print("\n2️⃣ Getting database information...")
        db_info = self.get_db_info()
        if db_info:
            print(f"   Dimensions: {db_info.get('dimensions', 'Unknown')}")
            print(f"   Vector count: {db_info.get('vector_count', 'Unknown')}")
            print(f"   Approximate search: {db_info.get('use_approximate', 'Unknown')}")
        
        # 3. Test embedding generation and check dimensions
        print("\n3️⃣ Testing embedding generation...")
        test_text = "This is a test sentence for embedding generation"
        embedding = self.generate_embedding(test_text)
        if embedding:
            embedding_dims = len(embedding)
            print(f"✓ Generated embedding with {embedding_dims} dimensions")
            
            # Check if dimensions match database
            if db_info and db_info.get('dimensions') != embedding_dims:
                print(f"⚠️  DIMENSION MISMATCH DETECTED!")
                print(f"   Database dimensions: {db_info.get('dimensions')}")
                print(f"   Embedding dimensions: {embedding_dims}")
                print(f"   Please restart your server with: ./build/vector_db_server --dimensions {embedding_dims}")
                return False
        else:
            print("❌ Failed to generate embedding")
            return False
        
        # 4. Populate database
        print("\n4️⃣ Populating database with sample data...")
        if not self.populate_database():
            print("❌ Failed to populate database")
            return False
        
        # 5. Test search
        print("\n5️⃣ Testing search functionality...")
        search_text = "programming and software development"
        print(f"Searching for: '{search_text}'")
        search_embedding = self.generate_embedding(search_text)
        if search_embedding:
            results = self.search_vectors(search_embedding, k=3)
            if results and results.get('results'):
                print(f"✓ Search returned {len(results['results'])} results")
                for i, result in enumerate(results['results'][:2], 1):
                    print(f"   {i}. {result.get('key')}: {result.get('metadata', '')[:60]}...")
            else:
                print("❌ Search failed or returned no results")
        
        print("\n✅ Test suite completed successfully!")
        return True

def main():
    """Main function"""
    print("Vector Database Test Script")
    print("=" * 40)
    
    try:
        # Initialize tester
        tester = VectorDBTester()
        
        # Quick dimension check first
        print("🔍 Checking embedding dimensions...")
        test_embedding = tester.generate_embedding("test")
        if test_embedding:
            print(f"Qwen model produces {len(test_embedding)}-dimensional embeddings")
        
        # Run test suite
        success = tester.run_test_suite()
        
        if success:
            # Start interactive query mode
            tester.interactive_query()
        else:
            print("\n❌ Test suite failed")
            print("\n💡 Common solutions:")
            print("1. Make sure your server is running:")
            print("   make run-server")
            print("2. If you get dimension mismatch, restart server with correct dimensions:")
            if test_embedding:
                print(f"   ./build/vector_db_server --dimensions {len(test_embedding)}")
            print("3. Check that NEBIUS_API_KEY is set correctly")
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Make sure to set your NEBIUS_API_KEY environment variable:")
        print("export NEBIUS_API_KEY='your_api_key_here'")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()