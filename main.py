import os
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from supabase import create_client, Client as SupabaseClient
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("SUPABASE_URL or SUPABASE_SERVICE_KEY not found in environment variables")
    raise ValueError("Supabase credentials are required")

app = FastAPI(title="File Storage Service", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
supabase: SupabaseClient = None

@app.on_event("startup")
async def startup_event():
    global supabase
    try:
        logger.info("Initializing Supabase client...")
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "client_ready": True,
        "supabase_ready": supabase is not None
    }

@app.post("/generate/")
async def store_and_share_file(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    receiver_uids: str = Form(...)
):
    """Store file in Supabase and share with receivers via Firebase"""
    temp_file_path = None
    
    try:
        # File validation
        content_type = file.content_type or ""
        filename = file.filename or ""
        
        # Check if it's an image or video
        valid_image_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
        valid_video_types = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime']
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.mp4', '.avi', '.mov']
        
        is_image = any(content_type.startswith(ct) for ct in valid_image_types) or any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp'])
        is_video = any(content_type.startswith(ct) for ct in valid_video_types) or any(filename.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])
        
        if not (is_image or is_video):
            logger.warning(f"Invalid file - Content-Type: {content_type}, Filename: {filename}")
            raise HTTPException(status_code=400, detail="File must be an image or video")
        
        if len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        logger.info(f"Starting file storage for user {sender_uid}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Receivers: {receiver_uids}")
        logger.info(f"File info - Content-Type: {content_type}, Filename: {filename}")
        logger.info(f"File type detected: {'video' if is_video else 'image'}")

        # Create temp directory if it doesn't exist
        temp_dir = Path("/tmp")
        temp_dir.mkdir(exist_ok=True)

        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        file_extension = Path(filename).suffix or ('.mp4' if is_video else '.jpg')
        temp_file_path = temp_dir / f"{file_id}{file_extension}"

        # Save file
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"File saved to {temp_file_path}")

        # Validate file size (optional)
        file_size = temp_file_path.stat().st_size
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")

        # Check if Supabase client is available
        if supabase is None:
            raise HTTPException(status_code=503, detail="Storage service not available")

        # Upload file to Supabase storage
        file_url = await _upload_file_to_supabase(str(temp_file_path), sender_uid, is_video)
        
        logger.info(f"File uploaded to Supabase: {file_url}")

        # Save chat messages to Firebase for each receiver
        receiver_list = [uid.strip() for uid in receiver_uids.split(",") if uid.strip()]
        await _save_chat_messages_to_firebase(sender_uid, receiver_list, file_url, prompt, is_video)

        return JSONResponse({
            "success": True,
            "video_url" if is_video else "image_url": file_url,
            "sender_uid": sender_uid,
            "receiver_uids": receiver_list
        })

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error storing file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store file: {str(e)}"
        )
    
    finally:
        # Cleanup temporary file
        if temp_file_path and Path(temp_file_path).exists():
            try:
                Path(temp_file_path).unlink()
                logger.info(f"Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file_path}: {e}")

async def _upload_file_to_supabase(local_file_path: str, sender_uid: str, is_video: bool) -> str:
    """Upload file to Supabase storage and return public URL"""
    try:
        file_path = Path(local_file_path)
        if not file_path.exists():
            raise Exception(f"File not found: {local_file_path}")

        # Generate unique filename for Supabase storage
        file_id = str(uuid.uuid4())
        file_extension = file_path.suffix
        storage_bucket = "videos" if is_video else "images"
        storage_path = f"{storage_bucket}/{sender_uid}/{file_id}{file_extension}"

        # Read file
        with open(file_path, "rb") as file_data:
            file_content = file_data.read()

        logger.info(f"Uploading file to Supabase: {storage_path}")

        # Upload to Supabase storage
        try:
            result = supabase.storage.from_(storage_bucket).upload(
                path=storage_path,
                file=file_content,
                file_options={
                    "content-type": "video/mp4" if is_video else "image/jpeg",
                    "cache-control": "3600"
                }
            )
            logger.info(f"Upload result: {result}")
            
        except Exception as upload_error:
            logger.error(f"Upload failed: {upload_error}")
            raise Exception(f"Supabase upload failed: {upload_error}")

        # Get public URL
        try:
            url_result = supabase.storage.from_(storage_bucket).get_public_url(storage_path)
            logger.info(f"Generated public URL: {url_result}")
            
            if not url_result:
                raise Exception("Failed to get public URL")
            
            return url_result
            
        except Exception as url_error:
            logger.error(f"Failed to get public URL: {url_error}")
            raise Exception(f"Failed to get public URL: {url_error}")

    except Exception as e:
        logger.error(f"Failed to upload file to Supabase: {e}")
        raise Exception(f"Storage upload failed: {str(e)}")

async def _save_chat_messages_to_firebase(sender_uid: str, receiver_list: list, file_url: str, prompt: str, is_video: bool):
    """Save chat messages with file URL to Firebase for each receiver"""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        from datetime import datetime
        import pytz
        
        # Initialize Firebase Admin if not already done
        if not firebase_admin._apps:
            try:
                # Use the specified service account file path
                cred = credentials.Certificate("/etc/secrets/services")
                firebase_admin.initialize_app(cred)
            except Exception as e:
                logger.error(f"Failed to initialize Firebase with service account: {e}")
                raise Exception("Firebase initialization failed")
        
        db = firestore.client()
        
        # Current timestamp with timezone
        ist = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(ist)
        
        media_type = "video" if is_video else "image"
        logger.info(f"Saving {media_type} messages to Firebase for {len(receiver_list)} receivers")
        
        for receiver_id in receiver_list:
            if not receiver_id:  # Skip empty receiver IDs
                continue
                
            try:
                logger.info(f"Processing message for receiver: {receiver_id}")
                
                # Create message document with all required fields
                message_data = {
                    "senderId": sender_uid,
                    "receiverId": receiver_id,
                    "text": prompt,  # The prompt as message text
                    "messageType": media_type,  # Message type - video or image
                    "timestamp": timestamp,
                    "isRead": False,
                    "createdAt": timestamp,
                    "updatedAt": timestamp,
                    "mediaType": media_type,  # Explicit media type
                    "mediaStatus": "uploaded"  # Status of media (uploaded, processing, failed)
                }
                
                # Add URL field based on media type
                if is_video:
                    message_data["videoUrl"] = file_url
                    message_data["hasVideo"] = True
                else:
                    message_data["imageUrl"] = file_url
                    message_data["hasImage"] = True
                
                # Save message to chats/{receiver_id}/messages/ collection
                doc_ref = db.collection("chats").document(receiver_id).collection("messages").add(message_data)
                message_id = doc_ref[1].id
                logger.info(f"{media_type.capitalize()} message saved to chats/{receiver_id}/messages/ with ID: {message_id}")
                
                # Also save to sender's chat collection for their own reference
                doc_ref_sender = db.collection("chats").document(sender_uid).collection("messages").add(message_data)
                sender_message_id = doc_ref_sender[1].id
                logger.info(f"{media_type.capitalize()} message saved to chats/{sender_uid}/messages/ with ID: {sender_message_id}")
                
                # Create or update chat document
                chat_participants = sorted([sender_uid, receiver_id])
                chat_id = f"{chat_participants[0]}_{chat_participants[1]}"
                
                # Updated chat data with media-specific fields
                chat_data = {
                    "participants": [sender_uid, receiver_id],
                    "participantIds": chat_participants,
                    "lastMessage": prompt,
                    "lastMessageType": media_type,
                    "lastMessageTimestamp": timestamp,
                    "lastSenderId": sender_uid,
                    "lastMediaType": media_type,
                    "updatedAt": timestamp,
                    "unreadCount": {
                        receiver_id: firestore.Increment(1)
                    }
                }
                
                # Add URL field based on media type
                if is_video:
                    chat_data["lastVideoUrl"] = file_url
                    chat_data["hasUnreadVideo"] = True
                else:
                    chat_data["lastImageUrl"] = file_url
                    chat_data["hasUnreadImage"] = True
                
                # Create chat if it doesn't exist, or update if it does
                chat_ref = db.collection("chats").document(chat_id)
                
                # Check if chat exists
                chat_doc = chat_ref.get()
                if chat_doc.exists:
                    # Update existing chat
                    update_data = {
                        "lastMessage": prompt,
                        "lastMessageType": media_type,
                        "lastMessageTimestamp": timestamp,
                        "lastSenderId": sender_uid,
                        "lastMediaType": media_type,
                        "updatedAt": timestamp,
                        f"unreadCount.{receiver_id}": firestore.Increment(1)
                    }
                    
                    # Add URL field based on media type
                    if is_video:
                        update_data["lastVideoUrl"] = file_url
                        update_data["hasUnreadVideo"] = True
                    else:
                        update_data["lastImageUrl"] = file_url
                        update_data["hasUnreadImage"] = True
                    
                    chat_ref.update(update_data)
                    logger.info(f"Updated existing chat with {media_type}: {chat_id}")
                else:
                    # Create new chat
                    chat_data["createdAt"] = timestamp
                    chat_data["unreadCount"] = {
                        sender_uid: 0,
                        receiver_id: 1
                    }
                    chat_ref.set(chat_data)
                    logger.info(f"Created new chat with {media_type}: {chat_id}")
                
            except Exception as e:
                logger.error(f"Failed to save {media_type} message for receiver {receiver_id}: {e}")
                continue  # Continue with other receivers even if one fails
        
        logger.info(f"Successfully saved all {media_type} messages with URLs to Firebase")
        
    except Exception as e:
        logger.error(f"Failed to save chat messages to Firebase: {e}", exc_info=True)
        # Don't raise exception here - file storage was successful
        # Just log the error and continue

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=600,  # 5 minutes keep alive
        timeout_graceful_shutdown=30
    )
