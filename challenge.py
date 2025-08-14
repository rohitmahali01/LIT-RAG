# challenge.py (Version 12.2.0 - Final with Dynamic PDF Landmark Lookup)
#
# Contains standalone handlers for special, multi-step challenges.
# This keeps the main application logic clean and focused on the RAG pipeline.

import httpx
import asyncio
import google.generativeai as genai
from typing import Optional, Dict, Any

# This dictionary will be populated from the main app's startup event.
# It's a simple way to share the initialized model without circular dependencies.
models: Dict[str, Any] = {}

async def fetch_dynamic_token(url: str) -> str:
    """
    Fetches the live secret token by parsing it from the HTML response.
    """
    print("[TOKEN FETCHER] Fetching and parsing dynamic secret token...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            html_content = response.text
            try:
                part_after_tag = html_content.split('<div id="token">')[1]
                token = part_after_tag.split('</div>')[0].strip()
                print(f"[TOKEN FETCHER] Successfully parsed token.")
                return token
            except IndexError:
                print("[TOKEN FETCHER] Error: Could not find '<div id=\"token\">' in the HTML response.")
                return "Error: Failed to parse token from HTML."
    except Exception as e:
        print(f"[TOKEN FETCHER] An error occurred while fetching the token: {e}")
        return "Error: Could not fetch the dynamic token."

async def find_landmark_in_pdf_with_llm(pdf_bytes: bytes, city: str) -> Optional[str]:
    """
    Makes a targeted multi-modal LLM call to find a landmark for a specific city
    by analyzing the provided PDF file bytes.
    """
    print(f"🤖 [FLIGHT PUZZLE] Calling LLM to find the landmark for '{city}' in the provided PDF file...")
    
    prompt = f"""
    You are a data lookup specialist. In the attached PDF file, find the landmark for the following city: {city}.

    Instructions:
    1. Scan the document for all entries related to the city: '{city}'.
    2. If the city appears more than once, you MUST use the landmark from its LAST appearance in the document.
    3. Your output MUST be ONLY the name of the landmark and nothing else. Do not include any explanatory text, greetings, or punctuation.
    """

    pdf_file_for_llm = {"mime_type": "application/pdf", "data": pdf_bytes}

    try:
        # Use the globally configured model from the main app
        if "generation_model" not in models:
            raise ValueError("Generation model not initialized in the 'models' dictionary.")
            
        model = models["generation_model"]
        response = await model.generate_content_async([prompt, pdf_file_for_llm])
        landmark = response.text.strip()
        print(f"✅ [FLIGHT PUZZLE] LLM lookup successful. Found landmark: '{landmark}'")
        return landmark
    except Exception as e:
        print(f"❌ [FLIGHT PUZZLE] An unexpected error occurred during the LLM call: {e}")
        return None

async def solve_flight_puzzle(pdf_url: str) -> Optional[str]:
    """
    Solves the multi-step API puzzle by downloading the PDF and using a multi-modal LLM
    to find the correct landmark, then retrieving the flight number.
    """
    print("[FLIGHT PUZZLE] Special challenge detected. Running LLM-based flight puzzle solver...")
    try:
        async with httpx.AsyncClient() as client:
            # 1. Download the PDF file
            print(f"[FLIGHT PUZZLE] Step 1: Downloading PDF from {pdf_url}...")
            pdf_response = await client.get(pdf_url, timeout=60.0)
            pdf_response.raise_for_status()
            pdf_content_bytes = pdf_response.content
            print(f"[FLIGHT PUZZLE] ✅ PDF downloaded successfully.")

            # 2. Get the Secret City
            print("[FLIGHT PUZZLE] Step 2: Fetching the secret city...")
            city_response = await client.get("https://register.hackrx.in/submissions/myFavouriteCity")
            city_response.raise_for_status()
            response_data = city_response.json()
            my_city = response_data['data']['city']
            print(f"[FLIGHT PUZZLE] ✅ Secret city is: '{my_city}'")

            # 3. Call the LLM with the downloaded PDF bytes and the secret city
            my_landmark = await find_landmark_in_pdf_with_llm(pdf_content_bytes, my_city)
            
            if not my_landmark:
                print("[FLIGHT PUZZLE] ❌ Solver failed: Could not find the landmark using the LLM.")
                return "Error: Could not determine landmark from PDF."
            
            # 4. Determine Flight Path and Get Final Number
            print(f"[FLIGHT PUZZLE] Step 3: Landmark is '{my_landmark}'. Determining flight URL.")
            base_url = "https://register.hackrx.in/teams/public/flights/"
            flight_url = ""
            if my_landmark == "Gateway of India": flight_url = base_url + "getFirstCityFlightNumber"
            elif my_landmark == "Taj Mahal": flight_url = base_url + "getSecondCityFlightNumber"
            elif my_landmark == "Eiffel Tower": flight_url = base_url + "getThirdCityFlightNumber"
            elif my_landmark == "Big Ben": flight_url = base_url + "getFourthCityFlightNumber"
            else: flight_url = base_url + "getFifthCityFlightNumber"
            
            print(f"[FLIGHT PUZZLE] Step 4: Requesting final flight number from: {flight_url}")
            flight_response = await client.get(flight_url)
            flight_response.raise_for_status()

            flight_data = flight_response.json()
            flight_number = flight_data.get('data', {}).get('flightNumber')

            if flight_number is None:
                flight_number = "API did not return a flight number."

            print(f"[FLIGHT PUZZLE] Step 5: Final flight number is '{flight_number}'")
            return str(flight_number)

    except Exception as e:
        print(f"[FLIGHT PUZZLE] ❌ An error occurred: {e}")
        return "Error occurred while solving flight puzzle."
