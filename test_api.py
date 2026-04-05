import requests
import json

# আপনার ডেপ্লয় করা API URL
API_URL = "http://eoo88swg0kswk4wgkkskkook.72.62.196.104.sslip.io/chat"

def test_gemini_api():
    print("--- Gemini API Test Start ---")
    
    # টেস্ট মেসেজ
    payload = {
        "message": "হাই, আমি একটি টেস্ট মেসেজ পাঠাচ্ছি। তুমি কি আমাকে শুনতে পাচ্ছ?"
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    try:
        print(f"Sending request to: {API_URL}...")
        response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print("\n[SUCCESS] Response received:")
            print(f"Gemini says: {result.get('text')}")
            print(f"Conversation ID: {result.get('conversation_id')}")
        else:
            print(f"\n[FAILED] Error code: {response.status_code}")
            print(f"Message: {response.text}")
            
    except Exception as e:
        print(f"\n[ERROR] Could not connect to API: {str(e)}")

if __name__ == "__main__":
    test_gemini_api()
