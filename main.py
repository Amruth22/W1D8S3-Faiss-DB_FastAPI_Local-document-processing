#!/usr/bin/env python3
"""
Main entry point for the PDF RAG Pipeline API
"""

import uvicorn
import argparse
import sys
import os
from dotenv import load_dotenv

def main():
    """Main function to start the FastAPI server"""
    # Get the absolute path of the parent directory (project root)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Add the project root to the path so we can import modules
    sys.path.insert(0, project_root)
    
    # Load environment variables from .env file in the project root
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print(f"Warning: .env file not found at {env_path}")
    
    parser = argparse.ArgumentParser(description="PDF RAG Pipeline API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the FastAPI server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port for the FastAPI server (default: 8001)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"Starting PDF RAG Pipeline API Server...")
    print(f"Access the API documentation at: http://localhost:{args.port}/docs")
    print(f"Access the Swagger UI at: http://localhost:{args.port}/docs")
    print(f"Access the ReDoc documentation at: http://localhost:{args.port}/redoc")
    
    # Start the FastAPI server
    # Use absolute path for the module
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()