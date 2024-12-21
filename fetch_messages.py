import requests
import json

# Replace these with your GroupMe API Token and Group ID
API_TOKEN = ""
GROUP_ID = ""

# GroupMe API endpoint for messages
BASE_URL = "https://api.groupme.com/v3/groups"

def fetch_messages(group_id, api_token, limit=100, before_id=None):
    """
    Fetch messages from a GroupMe group.
    
    Parameters:
        group_id (str): Group ID to fetch messages from.
        api_token (str): GroupMe API Token.
        limit (int): Number of messages to fetch (max 100).
        
    Returns:
        list: A list of message objects.
    """
    params = {
        "token": api_token,
        "limit": limit
    }

    if before_id:
        params["before_id"] = before_id

    response = requests.get(f"{BASE_URL}/{group_id}/messages", params=params)

    if response.status_code == 200:
        return response.json()["response"]["messages"]
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

def save_to_json(data, filename):
    """
    Save data to a JSON file.
    
    Parameters:
        data (dict): Data to save.
        filename (str): File name for the JSON file.
    """
    with open(f"msgs/{filename}", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    all_messages = []
    fetched = True
    before_id = None
    output_file = f"{GROUP_ID}_messages.json"

    print("Fetching messages...")
    while fetched:
        messages = fetch_messages(GROUP_ID, API_TOKEN, before_id=before_id)

        print(f"Fetched {len(messages)} messages...")
        if messages:
            all_messages.extend(messages)
            # Save to JSON file
            save_to_json(all_messages, output_file)
            # save our oldest messages index to work back from 
            before_id = messages[-1]["id"]  
            print(f"Messages saved to {output_file}")
        else:
            fetched = False  # Stop if no more messages

    print(f"Fetched a total of {len(all_messages)} messages.")

if __name__ == "__main__":
    main()
