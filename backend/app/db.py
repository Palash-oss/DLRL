from motor.motor_asyncio import AsyncIOMotorClient
import os

# Using local MongoDB by default
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URL)
db = client.acrux_db

async def save_analysis(user_id, data):
    collection = db.analysis_history
    data["user_id"] = user_id
    await collection.insert_one(data)

async def get_user_history(user_id):
    collection = db.analysis_history
    cursor = collection.find({"user_id": user_id}).sort("_id", -1)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results

async def init_user_profile(user_id, name, email, phone):
    collection = db.users
    existing = await collection.find_one({"user_id": user_id})
    if not existing:
        user_data = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "phone": phone,
            "credits": 200,
            "created_at": __import__("datetime").datetime.now().isoformat()
        }
        await collection.insert_one(user_data)
        return user_data
    return existing

async def get_user_profile(user_id):
    collection = db.users
    user = await collection.find_one({"user_id": user_id})
    if user:
        user["_id"] = str(user["_id"])
    return user

async def deduct_credits(user_id, amount=1):
    collection = db.users
    await collection.update_one(
        {"user_id": user_id, "credits": {"$gte": amount}},
        {"$inc": {"credits": -amount}}
    )
