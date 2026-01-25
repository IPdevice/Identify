import os
import shutil
import argparse
import fcntl
import subprocess
from pathlib import Path
import time
import re
import json
from step.clean_rules_1 import clean1, load_rules_with_lock, save_rules_with_lock
from step.extract_rules import DeviceClassificationAnalyzer
from rules_growth_tracker import record_batch_processing
try:
    import tiktoken
except ModuleNotFoundError:
    tiktoken = None

MILVUS_DB_PATH = "./milvus.db"
MODEL_CACHE_DIR = "./cache/"
MAX_CLEANUP_RETRIES = 3
CLEANUP_DELAY = 2



def emergency_cleanup():
    print("Performing emergency resource cleanup...")

    for attempt in range(MAX_CLEANUP_RETRIES):
        print(f"Cleanup attempt {attempt+1}/{MAX_CLEANUP_RETRIES}")

        # 1. Force kill milvus process
        try:
            subprocess.run("pkill -f milvus", shell=True)
            time.sleep(1)
        except Exception as e:
            print(f"Error terminating process: {e}")

        # 2. Delete cache
        try:
            if Path(MODEL_CACHE_DIR).exists():
                shutil.rmtree(MODEL_CACHE_DIR)
            Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            print(f"Cache directory {MODEL_CACHE_DIR} deleted and recreated")
        except Exception as e:
            print(f"Error deleting cache: {e}")

        # 3. Delete lock file
        try:
            lock_file = Path(MILVUS_DB_PATH) / "milvus.lock"
            if lock_file.exists():
                lock_file.unlink()
                print(f"Lock file deleted: {lock_file}")
        except Exception as e:
            print(f"Error deleting lock file: {e}")

        # 4. Attempt to acquire lock for verification
        try:
            lock_file.touch(exist_ok=True)
            with open(lock_file, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print("Successfully acquired Milvus file lock, cleanup completed")
                fcntl.flock(f, fcntl.LOCK_UN)
                return True
        except BlockingIOError:
            print("Lock still held, retrying...")
        except Exception as e:
            print(f"Error verifying lock: {e}")

        time.sleep(CLEANUP_DELAY)

    print("Warning: Cleanup not fully successful")
    return False


import json
import asyncio
import aiofiles
import hashlib
import time
import subprocess  # Only keep for executing shell commands
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import requests
import traceback
from hmac import new as hmac_new
from crawl_content_async import search
from deepsearcher_py import *
from openai import AsyncOpenAI
import httpx

# Configuration
from config import API_KEY_OPENAI, BASE_URL_OPENAI
CACHE_DIR = Path("./cache")
BATCH_SIZE=1 # Batch processing size
MAX_RETRIES = 2  # Maximum retry attempts (reduce to improve speed)
CRAWLER_TIMEOUT=3
SUB_CLASS_DIR = Path("./subclass")
MILVUS_DB_PATH = "./milvus.db"  # Milvus database path
MODEL_CACHE_DIR = "./cache/"  # Model cache directory

# Token limit configuration
MAX_TOKENS = 270000  # Maximum token limit
MIN_INPUT_LENGTH = 1  # Minimum input length

# Sample validation configuration
SAMPLE_VALIDATION_THRESHOLD = 0.5  # Matching ratio threshold for sample validation (0.5 = 50%)
                                    # Rules with matching ratio exceeding this threshold will be filtered out

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Declare client variables, initialize within async context
client_deepseek = None
client_openai = None

# Sample cache and rule genericity evaluation cache
_samples_cache = None
_samples_cache_file = None
_device_string_cache = {}  # {device_id: device_string}

def get_cache_key(data: str) -> str:
    """Generate cache key"""
    return hashlib.md5(data.encode('utf-8')).hexdigest()

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count token number of text - simplified version, directly use character count"""
    if not text:
        return 0
    # Directly use character count as token count to avoid complex tiktoken calculation
    return len(text)

def validate_and_truncate_input(input_text: str, max_tokens: int = MAX_TOKENS, min_length: int = MIN_INPUT_LENGTH) -> str:
    """Validate and truncate input text to meet length requirements - simplified version"""
    if not input_text or not isinstance(input_text, str):
        raise ValueError("Input text cannot be empty")
    
    # Remove leading/trailing whitespace
    input_text = input_text.strip()
    
    if len(input_text) < min_length:
        raise ValueError(f"Input text is too short, requires at least {min_length} characters")
    
    # Directly use character count limit to avoid complex token calculation
    if len(input_text) > max_tokens:
        return input_text[:max_tokens]
    
    return input_text

def validate_messages(messages: list, max_tokens: int = MAX_TOKENS) -> list:
    """Validate message list to ensure total token count does not exceed limit - simplified version"""
    total_content = ""
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg:
            total_content += str(msg["content"]) + "\n"
    
    if not total_content.strip():
        raise ValueError("Message content cannot be empty")
    
    # Directly use character count limit to avoid complex token calculation
    if len(total_content) > max_tokens:
        # If exceeded, truncate user message content
        truncated_messages = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg and msg.get("role") == "user":
                # Only truncate user messages
                truncated_content = validate_and_truncate_input(str(msg["content"]), max_tokens)
                truncated_messages.append({
                    "role": msg["role"],
                    "content": truncated_content
                })
            else:
                truncated_messages.append(msg)
        return truncated_messages
    
    return messages

async def read_cache(key: str) -> Optional[str]:
    """Read cache"""
    cache_path = CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        async with aiofiles.open(cache_path, 'r', encoding='utf-8') as f:
            return await f.read()
    return None

async def write_cache(key: str, data: str) -> None:
    """Write cache"""
    cache_path = CACHE_DIR / f"{key}.json"
    async with aiofiles.open(cache_path, 'w', encoding='utf-8') as f:
        await f.write(data)

async def init_clients():
    """Initialize async clients (within event loop)"""
    global client_deepseek, client_openai
    
    # Create shared HTTP client, compatible with OpenAI SDK requirements
    async_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=10),
        timeout=httpx.Timeout(30.0)
    )
    
    client_deepseek = AsyncOpenAI(
        api_key=API_KEY_OPENAI,
        base_url=BASE_URL_OPENAI,
        http_client=async_client
    )

    client_openai = AsyncOpenAI(
        api_key=API_KEY_OPENAI,
        base_url=BASE_URL_OPENAI,
        http_client=async_client
    )

async def safe_crawler(query: str) -> dict:
    """Safely call synchronous crawler: fix asyncio.wait_for parameter error"""
    start_time = time.time()
    try:
        loop = asyncio.get_running_loop()
        
        print(f"[Crawler Start] Query keyword: {query}")
        
        # Wrap synchronous crawler call
        future = loop.run_in_executor(None, lambda: search(query))
        
        # Fix: Remove loop parameter, catch timeout exception first
        try:
            result = await asyncio.wait_for(future, timeout=CRAWLER_TIMEOUT)
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"[Crawler Timeout] Query keyword '{query}', elapsed: {elapsed:.2f}s (timeout limit: {CRAWLER_TIMEOUT}s)")
            # Attempt to cancel task (though executor tasks may not be truly cancelable)
            if not future.done():
                future.cancel()
            return {"error": "crawler_timeout", "query": query, "timeout": CRAWLER_TIMEOUT}
        
        elapsed = time.time() - start_time
        print(f"[Crawler Success] Query keyword: {query}, elapsed: {elapsed:.2f}s")
        
        if result:
            print(f"[Crawler Result Preview] {str(result)[:20]}...")
        
        if not isinstance(result, (dict, list)):
            print(f"Warning: Crawler returned unexpected format ({type(result)})")
            return {"error": "crawler_return_invalid_format", "content": str(result)}
        
        return result
    
    except requests.exceptions.RequestException as e:
        elapsed = time.time() - start_time
        error_details = {
            "type": type(e).__name__,
            "message": str(e),
            "query": query,
            "elapsed": f"{elapsed:.2f}s"
        }
        print(f"[Crawler Network Error] {json.dumps(error_details, ensure_ascii=False)}")
        return {"error": f"crawler_network_error_{type(e).__name__}", "detail": str(e), "query": query}
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Crawler Unknown Error] Query: '{query}', elapsed: {elapsed:.2f}s")
        traceback.print_exc()
        return {
            "error": "crawler_unknown_error", 
            "detail": str(e), 
            "query": query,
            "traceback": traceback.format_exc()
        }

async def search_agent(key_word: str) -> str:
    """Search agent with cache and safe crawler call"""
    cache_key = get_cache_key(f"search_{key_word}")
    cached_result = await read_cache(cache_key)
    
    if cached_result:
        return cached_result
    
    try:
        # Use safe async crawler call
        result = await safe_crawler(key_word)

        # --- Modification ---
        if isinstance(result, (dict, list)):
            raw_str = json.dumps(result, ensure_ascii=False, indent=4)  # Convert to string
        else:
            raw_str = str(result)

        summary = await summarize_json_content(raw_str)
        response = f"Detailed explanation for {key_word}: {summary}"
        
        # Save cache
        await write_cache(cache_key, response)
        return response
    except Exception as e:
        print(f"Error searching {key_word}: {str(e)}")
        return f"When classifying the device, focus on: {key_word}"


async def data_extract(input_text: str) -> Tuple[str, str]:
    """Data extraction agent with retry mechanism"""
    # Ensure clients are initialized
    global client_deepseek
    if not client_deepseek:
        await init_clients()
    
    # Validate and truncate input
    try:
        validated_input = validate_and_truncate_input(input_text)
    except ValueError as e:
        print(f"Data extraction input validation failed: {e}")
        return '{"important_data": "Input validation failed"}', 'no'
        
    for attempt in range(MAX_RETRIES):
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": """
                    You are a network protocol analysis assistant, please extract standardized information from scan results. 
                    Output JSON format as follows:
                        {
                            "important_data": "Extract device owner, purpose, organizational background, or other identifiable information such as web page content (note to avoid extracting HTML format tags like `<href>`, `<script>`, etc.). 
                            Focus on the following information: organization name, email address, filing number, geographic location, device identification, declarations, etc., or any information that can point to the device owner, user, or operator. 
                            Pay special attention to directly identifying device attribution, purpose, and functional content. 
                            For keywords that may not be explicitly listed (such as device model, serial number, brand, etc.), please extract them as well. 
                            Later, I will perform further searches and analysis based on these keywords."
                        }
                """},
                {"role": "user", "content": validated_input}
            ]
            
            # Validate total message length
            validated_messages = validate_messages(messages)
            
            # Extract important data
            response1 = await client_deepseek.chat.completions.create(
                model="gpt-5-nano",
                messages=validated_messages
            )
            
            summary1 = response1.choices[0].message.content
            print(f"[DEBUG] Raw content returned by AI: {repr(summary1)}")
            print(f"[DEBUG] Content length: {len(summary1) if summary1 else 0}")
            
            if not summary1 or summary1.strip() == "":
                print(f"AI returned empty content, attempt {attempt+1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(0.1)  # Reduce backoff time to 0.1 seconds
                    continue
                else:
                    return '{"important_data": "Data extraction failed"}', 'no'
            
            # Validate JSON format
            try:
                parsed_json = parse_json_from_ai_response(summary1)
                print(f"[DEBUG] JSON parsed successfully: {parsed_json}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"AI returned non-JSON format, attempt {attempt+1}/{MAX_RETRIES}")
                print(f"[DEBUG] JSON parse error: {str(e)}")
                print(f"[DEBUG] Error content: {repr(summary1[:200])}...")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(0.1)  # Reduce backoff time to 0.1 seconds
                    continue
                else:
                    return '{"important_data": "Data extraction failed"}', 'no'
            
            # Extract new vocabulary
            messages2 = [
                {"role": "system", "content": """You are a proper noun recognition assistant. Extract **proper nouns you don't recognize** from the input (such as device manufacturers, domain names, software names, etc.), ignoring common protocols (such as http, https), terminology, and version numbers (such as nginx, Apache, Ubuntu, TLSv1.2, etc.).
                - If there are unknown nouns and multiple keywords, separate them with spaces, e.g.: word1 word2 word3;
                - If none, return the character "None";
                - Strictly prohibit outputting any additional text or explanations.
                """},
                {"role": "user", "content": validated_input}
            ]
            
            # Validate total message length
            validated_messages2 = validate_messages(messages2)
            
            response2 = await client_deepseek.chat.completions.create(
                model="gpt-5-nano",
                messages=validated_messages2
            )
            
            new_words = response2.choices[0].message.content
            if not new_words or new_words.strip() == "":
                new_words = "None"
            
            print('Completed extraction of new vocabulary')
            
            if 'None' not in new_words and 'not' not in new_words:
                return summary1, new_words
            else:
                return summary1, 'no'
                
        except Exception as e:
            print(f"Data extraction error (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(0.1)  # Reduce backoff time to 0.1 seconds
            
    return '{"important_data": "Data extraction failed"}', 'no'  # Return default value after multiple failed attempts

async def label_one_agent(input_text: str) -> str:
    """Primary label agent with retry mechanism"""
    # Validate and truncate input
    try:
        validated_input = validate_and_truncate_input(input_text)
    except ValueError as e:
        print(f"Primary label input validation failed: {e}")
        return '{"main_category": "Other", "reasoning_process": "Input validation failed"}'
    
    for attempt in range(MAX_RETRIES):
        try:
            messages = [
                {"role": "system", "content": """
                Please strictly follow the following rules for reasoning:
[Classification Rules]
Mainly based on device function, application scenario (what scenario the device is generally used in, such as IoT, industrial control, service, network infrastructure, terminal scenario) and keyword context, rather than simply relying on port numbers, protocols, or HTTP service characteristics.
Example: Open http/https ports ≠ necessarily a server. Need to reason based on title, path, Cookie, Header, certificate, manufacturer information, keyword fields, etc.
Example: Appearance of Niagara, Milesight_Industrial_Cellular_Routers, SCADA, Modbus, BACnet, Siemens, PLC → highly points to Industrial Control System (ICS), should be prioritized as ICS even if running on http.
Classification System (Six Major Categories):
- Network Infrastructure: Routing, switching, firewall, load balancing, gateway, WAP and other network supporting devices.
- Service Devices: Facing application layer, providing computing, storage, applications (such as Windows Server, DNS server, Mail server, FTP server, CDN service, virtualization platform, NAS).
- IoT Devices: Domoticz, cameras, IoT phone VoIP, printers, video/media/telephone/smart home devices such as Chromecast, facing home or office interaction.
- User Terminal Devices: Workstations, Mac OS X, VDCERPC (RPC Endpoint Mapper), Windows, VMware Workstation, Apple Remote Desktop, Android Debug Bridge (ADB),
PC, mobile phones, directly operated by users.
- Industrial Control Systems: Industrial production, energy, power, building automation control related equipment (such as Niagara Framework, SCADA systems, PLC, HMI, industrial routers, building control systems).
- Other: Only use when it is really impossible to classify, avoid as much as possible.
Reasoning Priority:
Priority to identify industry/domain keywords → function and application scenario.
Secondly consider service characteristics and manufacturer information.
Finally refer to port/protocol, but cannot be used as the sole basis for classification.
Special Notes:  
Mainly focus on the actual device purpose and function indicated by keywords.
If the description is a Web login page but contains characteristics of other types of devices (such as Niagara Login interface), classify as the corresponding category instead of general service device.
[Classification Process]
- Step-by-step analysis: manufacturer/protocol/title/path/field/keywords.
- Judge device function and application scenario.
- Output according to one of the above six categories.
[Output JSON Format]
{
  "main_category": "xxx",
  "reasoning_process": "Detailed explanation of how keyword fields and functions point to this category"
}
                """},
                {"role": "user", "content": validated_input}
            ]
            
            # Validate total message length
            validated_messages = validate_messages(messages)
            
            response = await client_deepseek.chat.completions.create(
                model="gpt-5-nano",
                messages=validated_messages
            )
            label_one = response.choices[0].message.content
            print(f"[DEBUG] Primary label AI response content: {repr(label_one)}")
            print(f"[DEBUG] Primary label content length: {len(label_one) if label_one else 0}")
            return label_one
        except Exception as e:
            print(f"Primary label classification error (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(0.1)  # Reduce backoff time to 0.1 seconds
    
    return '{"main_category": "Other", "reasoning_process": "Classification failed"}'

async def load_subclass(file_name: str) -> List[str]:
    """Load subclass list"""
    file_path = SUB_CLASS_DIR / file_name
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
        
        subclass = [line.strip() for line in content.split('\n') if line.strip()]
        subclass = list(set(subclass))  # Remove duplicates
        return subclass
    except Exception as e:
        print(f"Error loading subclass file {file_name}: {str(e)}")
        return []

async def infra_subclass_agent(input_text: str) -> str:
    """Network infrastructure subclass agent"""
    subclass = await load_subclass("network_infrastructure.txt")
    subclass_text = '\n'.join(subclass)
    
    # Validate and truncate input
    try:
        validated_input = validate_and_truncate_input(input_text)
    except ValueError as e:
        print(f"Network infrastructure subclass input validation failed: {e}")
        return '{"new_subclass_exists": "No", "subclass": "Other", "reasoning_process": "Input validation failed"}'
    
    try:
        messages = [
            {"role": "system", "content": f"""
            You are a network infrastructure device classification assistant. The input is a device description that has been determined to be [Network Infrastructure].

            Known subclass list (extensible):
            {subclass_text}

            [Classification Process]:
            1. Prioritize judging subclass based on device function and typical application scenarios, not just relying on name or web page characteristics.
            2. If it matches an existing subclass, output:
            {{
              "new_subclass_exists": "No",
              "subclass": "<matched_subclass>",
              "reasoning_process": "Explain why it matches this subclass"
            }}
            3. If it does not match any existing subclass, you must generate a new **general subclass**:
               - The new subclass must remain general, and direct use of specific device names such as Cisco, BigIP, FortiGate is not allowed.
               - The new subclass must be decoupled from existing subclasses and cover the main functions of the input device.
            4. The output JSON must be strictly formatted without extra text.
            5. Avoid misjudgment due to HTTP, port, domain name, IP and other words in the description. The core basis should be device function and application scenario.
            [Output JSON Format]:
            {{
              "new_subclass_exists": "Yes/No",
              "subclass": "<general_subclass_name>",
              "reasoning_process": "Explain why it matches this subclass or is classified as a new subclass, and give examples"
            }}
            """},
            {"role": "user", "content": validated_input}
        ]
        
        # Validate total message length
        validated_messages = validate_messages(messages)
        
        response = await client_openai.chat.completions.create(
            model="gpt-5-nano",
            messages=validated_messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Network infrastructure subclass classification error: {str(e)}")
        return '{"new_subclass_exists": "No", "subclass": "Other", "reasoning_process": "Classification failed"}'

async def service_subclass_agent(input_text: str) -> str:
    """Service device subclass agent"""
    subclass = await load_subclass("service_devices.txt")
    subclass_text = '\n'.join(subclass)
    
    # Validate and truncate input
    try:
        validated_input = validate_and_truncate_input(input_text)
    except ValueError as e:
        print(f"Service device subclass input validation failed: {e}")
        return '{"new_subclass_exists": "No", "subclass": "Other", "reasoning_process": "Input validation failed"}'
    
    try:
        messages = [
            {"role": "system", "content": f"""
            You are a service device subclass classification assistant. The input is a device description that has been determined to be [Service Devices].

            Known subclass list (extensible):
            {subclass_text}

            [Classification Process]:
            1. Prioritize judging subclass based on device function and typical application scenarios, not just relying on name or web page characteristics.
            2. If it matches an existing subclass, output:
            {{
              "new_subclass_exists": "No",
              "subclass": "<matched_subclass>",
              "reasoning_process": "Explain why it matches this subclass"
            }}
            3. If it does not match any existing subclass, you must generate a new **general subclass**:
               - The new subclass must remain general, and direct use of specific service names such as Windows Server, Synology, CDN is not allowed.
               - The new subclass must be decoupled from existing subclasses and cover the main functions of the input device.
            4. The output JSON must be strictly formatted without extra text.
            5. Avoid misjudgment as network service type due to HTTP, port, domain name, IP and other words in the description. The core basis should be device function and application scenario.
            [Output JSON Format]:
            {{
              "new_subclass_exists": "Yes/No",
              "subclass": "<general_subclass_name>",
              "reasoning_process": "Explain why it matches this subclass or is classified as a new subclass, and give examples"
            }}
            """},
            {"role": "user", "content": validated_input}
        ]
        
        # Validate total message length
        validated_messages = validate_messages(messages)
        
        response = await client_openai.chat.completions.create(
            model="gpt-5-nano",
            messages=validated_messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Service device subclass classification error: {str(e)}")
        return '{"new_subclass_exists": "No", "subclass": "Other", "reasoning_process": "Classification failed"}'

async def iot_subclass_agent(input_text: str) -> str:
    """IoT device subclass agent"""
    subclass = await load_subclass("IoT_devices.txt")
    subclass_text = '\n'.join(subclass)
    
    # Validate and truncate input
    try:
        validated_input = validate_and_truncate_input(input_text)
    except ValueError as e:
        print(f"IoT device subclass input validation failed: {e}")
        return '{"new_subclass_exists": "No", "subclass": "Other", "reasoning_process": "Input validation failed"}'
    
    try:
        messages = [
            {"role": "system", "content": f"""
            You are an IoT device subclass classification assistant. The input is a device description that has been determined to be [IoT Devices].

            Known subclass list (extensible):
            {subclass_text}

            [Classification Process]:
            1. Prioritize judging subclass based on device function and typical application scenarios, not just relying on web page characteristics (such as HTTP, login page, cookie).
            2. If it matches an existing subclass, output:
            {{
              "new_subclass_exists": "No",
              "subclass": "<matched_subclass>",
              "reasoning_process": "Explain why it matches this subclass"
            }}
            3. If it does not match any existing subclass, you must generate a new **general subclass**:
               - The new subclass needs to remain general and not too detailed.
               - Outputting "main category" or original device name as subclass is not allowed.
               - The new subclass must be decoupled from existing subclasses and cover the main functions of the input device.
            4. The output JSON must be strictly formatted without extra text.
            5. Note: Do not misjudge subclass due to web page, port, cookie and other words in the description, the core basis should be device function and application scenario.
            [Output JSON Format]:
            {{
              "new_subclass_exists": "Yes/No",
              "subclass": "<general_subclass_name>",
              "reasoning_process": "Explain why it matches this subclass or is classified as a new subclass, and give examples"
            }}
            """},
            {"role": "user", "content": validated_input}
        ]
        
        # Validate total message length
        validated_messages = validate_messages(messages)
        
        response = await client_openai.chat.completions.create(
            model="gpt-5-nano",
            messages=validated_messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"IoT device subclass classification error: {str(e)}")
        return '{"new_subclass_exists": "No", "subclass": "Other", "reasoning_process": "Classification failed"}'

async def terminal_subclass_agent(input_text: str) -> str:
    """User terminal device subclass agent"""
    subclass = await load_subclass("user_terminals.txt")
    subclass_text = '\n'.join(subclass)
    
    # Validate and truncate input
    try:
        validated_input = validate_and_truncate_input(input_text)
    except ValueError as e:
        print(f"User terminal device subclass input validation failed: {e}")
        return '{"new_subclass_exists": "No", "subclass": "Other", "reasoning_process": "Input validation failed"}'
    
    try:
        messages = [
            {"role": "system", "content": f"""
            You are a user terminal device subclass classification assistant. The input is a device description that has been determined to be [User Terminal Devices].

            Known subclass list (extensible):
            {subclass_text}

            [Classification Process]:
            1. Prioritize judging subclass based on device function and typical application scenarios, not just relying on web page characteristics (such as HTTP, login page, cookie).
            2. If it matches an existing subclass (i.e., in the known subclass list), output:
            {{
              "new_subclass_exists": "No",
              "subclass": "<matched_subclass>",
              "reasoning_process": "Explain why it matches this subclass"
            }}
            3. If it does not match any existing subclass, you must generate a new **general subclass**:
               - The new subclass needs to remain general and not too detailed.
               - Outputting "main category" or original device name as subclass is not allowed.
               - The new subclass must be decoupled from existing subclasses and cover the main functions of the input device.
            4. The output JSON must be strictly formatted without extra text.
            5. Note: Do not misjudge subclass due to web page, port, cookie and other words in the description, the core basis should be device function and application scenario.
            [Output JSON Format]:
            {{
              "new_subclass_exists": "Yes/No",
              "subclass": "<general_subclass_name>",
              "reasoning_process": "Explain why it matches this subclass or is classified as a new subclass, and give examples"
            }}
            """},
            {"role": "user", "content": validated_input}
        ]
        
        # Validate total message length
        validated_messages = validate_messages(messages)
        
        response = await client_openai.chat.completions.create(
            model="gpt-5-nano",
            messages=validated_messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"User terminal device subclass classification error: {str(e)}")
        return '{"new_subclass_exists": "No", "subclass": "Other", "reasoning_process": "Classification failed"}'

async def ics_subclass_agent(input_text: str) -> str:
    """Industrial control system subclass agent"""
    subclass = await load_subclass("ICSs.txt")
    subclass_text = '\n'.join(subclass)
    
    # Validate and truncate input
    try:
        validated_input = validate_and_truncate_input(input_text)
    except ValueError as e:
        print(f"Industrial control system subclass input validation failed: {e}")
        return '{"new_subclass_exists": "No", "subclass": "Other", "reasoning_process": "Input validation failed"}'
    
    try:
        messages = [
            {"role": "system", "content": f"""
            You are an industrial control system subclass classification assistant. The input is a device description that has been determined to be [Industrial Control Systems].

            Known subclass list (extensible):
            {subclass_text}

            [Classification Process]:
            1. Prioritize judging subclass based on device function and typical application scenarios, not just relying on web page characteristics (such as HTTP, login page, cookie).
            2. If it matches an existing subclass (i.e., in the known subclass list), output:
            {{
              "new_subclass_exists": "No",
              "subclass": "<matched_subclass>",
              "reasoning_process": "Explain why it matches this subclass"
            }}
            3. If it does not match any existing subclass, you must generate a new **general subclass**:
               - The new subclass needs to remain general and not too detailed. Outputting overly detailed new subclasses is not allowed, e.g., cannot output 'Industrial Firewall' but need to output 'Industrial Security Device', cannot output overly detailed ones like 'Building Automation Control System' but need general categories.
               - Outputting "main category" or "Industrial Control Systems" as subclass is not allowed.
               - The new subclass must be decoupled from existing subclasses and cover the main functions of the input device.
            4. Note: Do not misjudge as "Communication Network Device" due to network words such as login, web page, cookie, IP in the description. Only judge as Communication Network Device when gateway, router, serial server and other hardware are explicitly mentioned.
            [Output JSON Format]:
            {{
              "new_subclass_exists": "Yes/No",
              "subclass": "<general_subclass_name>",
              "reasoning_process": "Explain why it matches this subclass or is classified as a new subclass, and give examples"
            }}
            """},
            {"role": "user", "content": validated_input}
        ]
        
        # Validate total message length
        validated_messages = validate_messages(messages)
        
        response = await client_openai.chat.completions.create(
            model="gpt-5-nano",
            messages=validated_messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Industrial control system subclass classification error: {str(e)}")
        return '{"new_subclass_exists": "No", "subclass": "Other", "reasoning_process": "Classification failed"}'

async def label_subclass(big_class: str, input_text: str) -> str:
    """Automatically call corresponding subclass agent based on main category name"""
    print(f'label_subclass: {big_class}')
    if "Network Infrastructure" in big_class:
        print('Classified as Network Infrastructure')
        return await infra_subclass_agent(input_text)
    elif "Service Devices" in big_class:
        print('Classified as Service Devices')
        return await service_subclass_agent(input_text)
    elif "IoT Devices" in big_class:
        print('Classified as IoT Devices')
        return await iot_subclass_agent(input_text)
    elif "User Terminal Devices" in big_class:
        print('Classified as User Terminal Devices')
        return await terminal_subclass_agent(input_text)
    elif "Industrial Control Systems" in big_class:
        print('Classified as Industrial Control Systems')
        return await ics_subclass_agent(input_text)
    else:
        return '{"new_subclass_exists": "No", "subclass": "Unknown", "reasoning_process": "Unknown main category"}'

async def rule_extract_agent(raw_data: str, important_data: str, all_label: str) -> str:
    """Rule extraction agent"""
    # Validate and truncate input
    try:
        validated_raw_data = validate_and_truncate_input(raw_data)
        validated_important_data = validate_and_truncate_input(important_data)
        validated_all_label = validate_and_truncate_input(all_label)
    except ValueError as e:
        print(f"Rule extraction input validation failed: {e}")
        return '{"label": "", "keywords": []}'
    
    try:
        user_content = f'Raw data: {validated_raw_data}\nClassification category: {validated_all_label}'
        
        # Validate user content length again
        validated_user_content = validate_and_truncate_input(user_content)
        
        messages = [
            {"role": "system", "content": f"""
            You are a rule extraction assistant, used to identify keywords related to the specified label from network scan raw data segments.  
            Input:
            1. Raw scan data (JSON format), including IP, port, service, protocol, geographic information, ASN information, HTTP headers, Cookie, title, etc.
            2. Label information (main_category|subclass), such as:
               - main_category: Industrial Control Systems
               - subclass: Automation Systems
            Task:
            Extract keywords from raw data that can support classifying the device into the specified label, and give reasons for rule classification.  
            - Each record should include:
              1. Field source (raw data)
              2. Keyword (must be a single string, cannot contain JSON objects), extract at least 2 keywords, and cannot extract IP addresses
              3. Reason for rule classification (explain why this keyword supports classifying the device into this label)
              4. All keywords must exist in the raw data without any additional information
            - Only extract keywords and reasons for rule classification, no additional explanations needed.
            - Keywords must be simple string values, not JSON objects or containing special characters
            Output example:
            {{
              "label": "main_category|subclass",
              "keywords": [
                  {{"field_source": "header", "keyword": "Niagara Framework", "classification_reason": "This framework is used for building automation and industrial control systems, reflecting control and management functions"}},
                  {{"field_source": "cookie", "keyword": "niagara_origin_uri", "classification_reason": "Cookie identifier indicates the device runs Niagara Framework, supporting automation system functions"}}
              ]
            }}
            
            **Important: Keyword Specificity Requirements**
            - Each keyword must be **highly specific** and able to clearly distinguish device types, rather than fuzzy matching
            - Strictly prohibit extracting overly general keywords. Overly general keywords refer to keywords that appear in most or all devices and cannot effectively distinguish device types, such as:
              * General protocol names: http, https, tcp, udp, etc. (unless they are identifiers of specific protocols)
              * HTTP status codes: 200 OK, 302 Found, 401 Unauthorized, etc.
              * General HTTP header fields: Content-Type, Content-Length, Server, Date, etc.
              * General strings: yes, no, true, false, null, etc.
              * General security policies: SAMEORIGIN, X-Frame-Options, etc.
              * Port numbers: 80, 443, 8080, etc.
              * General category words: such as "Service Devices", "Application Service", "Network Devices", etc. (these words are too broad to distinguish specific devices)
            
            **Strict Requirements Specifically for "Service Devices|Application Service Devices" Label**:
            - If the label is "Service Devices|Application Service Devices", must extract **very specific** keywords, such as:
              * Specific software names (e.g., "Windows Server", "Apache", "Nginx", etc., but must be complete product names)
              * Specific service identifiers (e.g., "Microsoft-IIS", "nginx/1.18.0", etc.)
              * Specific application frameworks (e.g., "Django", "WordPress", etc.)
            - **Strictly prohibit** extracting the following general keywords (these will cause a large number of mis-matches):
              * General words such as "http", "https", "web", "server"
              * Category words such as "Service Devices", "Application Service"
              * HTTP status codes, general HTTP headers, etc.
            - If the raw data only contains general information (such as only http, port numbers, etc.), **do not extract rules** and return empty keyword list
            
            **Keyword Extraction Principles**:
            1. Prioritize extracting device manufacturer names (e.g., Siemens, Moxa, Niagara, etc.)
            2. Prioritize extracting device models or product names (e.g., MRD-405, Scalance M800, etc.)
            3. Prioritize extracting specific software or framework names (e.g., GoAhead-Webs, Niagara Framework, etc.)
            4. Prioritize extracting device-specific identifiers (e.g., specific cookie names, header values, etc.)
            5. When extracting keywords, carefully consider: Does this keyword appear in most devices? If yes, do not extract it.
            6. If the extracted keyword cannot effectively distinguish device types, it is better not to extract the rule.
            """},
            {"role": "user", "content": validated_user_content}
        ]
        
        # Validate total message length
        validated_messages = validate_messages(messages)
        
        response = await client_deepseek.chat.completions.create(
            model="gpt-5-nano",
            messages=validated_messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Rule extraction error: {str(e)}")
        return '{"label": "", "keywords": []}'

async def evaluate_keyword_genericness(keyword: str, label: str) -> Dict[str, Any]:
    """Use large model to evaluate if keyword is too generic
    
    Returns:
        {
            "is_generic": bool,  # Whether too generic
            "reason": str,       # Reason for judgment
            "confidence": float  # Confidence 0-1
        }
    """
    try:
        messages = [
            {"role": "system", "content": """
            You are a keyword genericness evaluation expert. Your task is to judge whether a keyword is too generic to effectively distinguish device types.

            Evaluation Criteria:
            1. **Characteristics of overly generic keywords**:
               - Appear in most or all devices (e.g., http, yes, true, etc.)
               - Cannot distinguish device types or functions (e.g., port numbers, general protocol names, etc.)
               - Are standard protocols, status codes, general fields, etc. (e.g., HTTP/1.1, 200 OK, Content-Type, etc.)
            
            2. **Characteristics of effective keywords**:
               - Device manufacturer names (e.g., Siemens, Moxa, Niagara, etc.)
               - Device models or product names (e.g., MRD-405, Scalance M800, etc.)
               - Specific software or framework names (e.g., GoAhead-Webs, Niagara Framework, etc.)
               - Device-specific identifiers or characteristics (e.g., specific cookie names, header values, etc.)
            
            3. **Context Consideration**:
               - Judge combined with label information: If the keyword is too generic for the label category, it should also be judged as generic
               - Example: For "Service Devices|Application Service Devices" label, "http" is generic; but for "Industrial Control Systems" label, "http" may not be generic (if the device actually uses http)
            
            Output JSON format:
            {
              "is_generic": true/false,
              "reason": "Detailed explanation of why this keyword is generic or not",
              "confidence": 0.0-1.0
            }
            """},
            {"role": "user", "content": f"""
            Please evaluate whether the following keyword is too generic:
            
            Keyword: {keyword}
            Label: {label}
            
            Please judge whether this keyword is too generic to effectively distinguish device types.
            """}
        ]
        
        response = await client_deepseek.chat.completions.create(
            model="gpt-5-nano",
            messages=messages
        )
        
        result_text = response.choices[0].message.content
        result = parse_json_from_ai_response(result_text)
        
        # Ensure return format is correct
        return {
            "is_generic": result.get("is_generic", False),
            "reason": result.get("reason", ""),
            "confidence": float(result.get("confidence", 0.5))
        }
    except Exception as e:
        print(f"Error evaluating keyword genericness: {str(e)}")
        # Conservative handling when error occurs: if cannot judge, default to not generic (avoid false filtering)
        return {
            "is_generic": False,
            "reason": f"Evaluation error: {str(e)}",
            "confidence": 0.0
        }

# Rule genericness evaluation cache (avoid repeated evaluation of the same rule)
_rule_genericness_cache: Dict[str, Dict[str, Any]] = {}

async def evaluate_rule_genericness(rule: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
    """Use large model to evaluate if the entire rule is too generic
    
    Args:
        rule: Rule to evaluate
        use_cache: Whether to use cache
    
    Returns:
        {
            "is_too_generic": bool,  # Whether rule is too generic
            "generic_keywords": List[str],  # List of overly generic keywords
            "reason": str,  # Reason for judgment
            "confidence": float  # Confidence
        }
    """
    try:
        label = rule.get('Label', rule.get('label', ''))
        keywords = rule.get('Keywords', rule.get('keywords', []))
        
        # Generate cache key (based on label and keywords)
        keyword_strings = []
        for keyword_item in keywords:
            if isinstance(keyword_item, dict):
                keyword_strings.append(str(keyword_item.get('keyword', '')))
            elif isinstance(keyword_item, str):
                keyword_strings.append(keyword_item)
            else:
                keyword_strings.append(str(keyword_item))
        
        # Check cache
        cache_key = f"{label}|||{sorted(keyword_strings)}"
        if use_cache and cache_key in _rule_genericness_cache:
            return _rule_genericness_cache[cache_key]
        
        keyword_list_str = ', '.join([f'"{kw}"' for kw in keyword_strings if kw])
        
        messages = [
            {"role": "system", "content": f"""
            You are a rule genericness evaluation expert. Your task is to judge whether a rule is too generic and will cause a large number of devices to be mis-matched.

            Evaluation Criteria (please strictly follow):
            1. **Characteristics of overly generic rules** (must meet all the following conditions to be judged as overly generic):
               - Keywords in the rule appear in **most or all device types** (e.g., http, yes, true, 200 OK, etc.)
               - Keywords in the rule **completely cannot distinguish device types** and will cause **almost all devices** to be mis-matched
               - Keywords in the rule are mainly **standard protocol names, HTTP status codes, general HTTP header fields**, etc. (e.g., http, https, Content-Type, 200 OK, etc.)
               - **Note**: If keywords contain device manufacturer names, device models, specific software names (such as GoAhead-Webs, Moxa, Siemens, etc.), even if they contain some general words, they should not be judged as overly generic
            
            2. **Characteristics of effective rules** (should be judged as effective rules in the following cases):
               - Keywords in the rule contain **device manufacturer names** (e.g., Siemens, Moxa, Niagara, Westermo, etc.)
               - Keywords in the rule contain **device models or product names** (e.g., MRD-405, Scalance M800, NB1601, etc.)
               - Keywords in the rule contain **specific software or framework names** (e.g., GoAhead-Webs, Niagara Framework, etc.)
               - Keywords in the rule contain **device-specific identifiers** (e.g., specific cookie names, header values, etc.)
               - Even if keywords contain some general words (such as "industrial", "router", etc.), as long as they also contain specific device information, they should not be judged as overly generic
            
            3. **Special Cases**:
               - If keywords contain category names (such as "Industrial Control Systems", "Communication Network Devices") but also contain specific device information, should be judged as effective rules
               - Only when keywords **consist entirely of general words** without any specific device information, judge as overly generic
            
            **Strict Evaluation Specifically for "Service Devices|Application Service Devices" Label**:
            - If the label is "Service Devices|Application Service Devices", must use **stricter standards** for evaluation
            - Such labels are very broad and prone to a large number of mis-matches, therefore:
              * If keywords only contain general protocols, status codes, general HTTP headers, etc., **must judge as overly generic** (confidence should be >= 0.8)
              * If keywords contain general words such as "http", "https", "web", "server" but no specific software/manufacturer information, **should judge as overly generic** (confidence should be >= 0.7)
              * Only judge as effective rules when keywords contain **specific software names, service identifiers, application frameworks** and other specific information
              * Example: "http", "200 OK" → overly generic (confidence 0.9)
              * Example: "nginx/1.18.0", "Microsoft-IIS" → effective rule (confidence 0.2)
            
            Output JSON format:
            {{
              "is_too_generic": true/false,
              "generic_keywords": ["keyword1", "keyword2"],  # List of overly generic keywords (empty list if is_too_generic is false)
              "reason": "Detailed explanation of why this rule is generic or not",
              "confidence": 0.0-1.0
            }}
            """},
            {"role": "user", "content": f"""
            Please evaluate whether the following rule is too generic:
            
            Label: {label}
            Keyword list: [{keyword_list_str}]
            
            Please judge whether this rule is too generic and will cause a large number of devices to be mis-matched.
            """}
        ]
        
        response = await client_deepseek.chat.completions.create(
            model="gpt-5-nano",
            messages=messages
        )
        
        result_text = response.choices[0].message.content
        result = parse_json_from_ai_response(result_text)
        
        evaluation_result = {
            "is_too_generic": result.get("is_too_generic", False),
            "generic_keywords": result.get("generic_keywords", []),
            "reason": result.get("reason", ""),
            "confidence": float(result.get("confidence", 0.5))
        }
        
        # Cache result
        if use_cache:
            _rule_genericness_cache[cache_key] = evaluation_result
            # Limit cache size to avoid memory overflow
            if len(_rule_genericness_cache) > 10000:
                # Delete oldest 50% of cache
                keys_to_remove = list(_rule_genericness_cache.keys())[:5000]
                for key in keys_to_remove:
                    del _rule_genericness_cache[key]
        
        return evaluation_result
    except Exception as e:
        print(f"Error evaluating rule genericness: {str(e)}")
        return {
            "is_too_generic": False,
            "generic_keywords": [],
            "reason": f"Evaluation error: {str(e)}",
            "confidence": 0.0
        }

def parse_json_from_ai_response(text: str) -> dict:
    """
    Parse JSON from AI response text, automatically handle format issues such as markdown code block markers
    
    Args:
        text: AI response text, may contain markdown code block markers
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        json.JSONDecodeError: If cannot parse to JSON
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text cannot be empty")
    
    # Step 1: Remove leading/trailing whitespace
    cleaned = text.strip()
    
    # Step 2: Try to remove markdown code block markers
    # Match ```json ... ``` or ``` ... ```
    # Remove leading ```json or ```
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    # Remove trailing ```
    cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = cleaned.strip()
    
    # Step 3: Try direct parsing
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Step 4: If direct parsing fails, try to extract JSON object
    # Find content between first { and last }
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = cleaned[first_brace:last_brace + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass
    
    # Step 5: If still failed, try to find content between first [ and last ] (handle JSON array)
    first_bracket = cleaned.find('[')
    last_bracket = cleaned.rfind(']')
    
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        json_candidate = cleaned[first_bracket:last_bracket + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass
    
    # If all methods fail, raise exception
    raise json.JSONDecodeError(
        f"Cannot parse JSON from text. First 200 characters of raw text: {text[:200]}",
        text,
        0
    )

def clean_and_validate_rules(rules_str: str) -> str:
    """Clean and validate rule string to ensure correct JSON format"""
    try:
        # First try to parse JSON (rule string may also contain markdown markers)
        try:
            rules_data = parse_json_from_ai_response(rules_str)
        except (json.JSONDecodeError, ValueError):
            # If parse_json_from_ai_response fails, try direct parsing
            rules_data = json.loads(rules_str)
        
        # Ensure label uses correct separator
        if 'label' in rules_data and isinstance(rules_data['label'], str):
            rules_data['label'] = rules_data['label'].replace('｜', '|')
        
        # Validate and clean keywords
        if 'keywords' in rules_data and isinstance(rules_data['keywords'], list):
            cleaned_keywords = []
            for keyword_item in rules_data['keywords']:
                if isinstance(keyword_item, dict):
                    # Ensure keyword is simple string
                    if 'keyword' in keyword_item:
                        keyword_value = keyword_item['keyword']
                        if isinstance(keyword_value, dict):
                            # If JSON object, extract key information
                            if 'name' in keyword_value:
                                keyword_item['keyword'] = keyword_value['name']
                            elif 'value' in keyword_value:
                                keyword_item['keyword'] = keyword_value['value']
                            else:
                                # Convert object to string
                                keyword_item['keyword'] = str(keyword_value)
                        elif not isinstance(keyword_value, str):
                            keyword_item['keyword'] = str(keyword_value)
                    
                    # Filter out keywords containing "JSON parsing failed"
                    if keyword_item.get('keyword') == 'JSON parsing failed':
                        print("Filtering out rule keywords containing 'JSON parsing failed'")
                        continue
                    
                    # Ensure field source exists
                    if 'field_source' not in keyword_item:
                        keyword_item['field_source'] = 'unknown'
                    
                    # Ensure classification reason exists
                    if 'classification_reason' not in keyword_item:
                        keyword_item['classification_reason'] = 'No reason provided'
                    
                    cleaned_keywords.append(keyword_item)
            
            rules_data['keywords'] = cleaned_keywords
        
        # Check if there are still valid keywords, return empty rule if none
        if not rules_data.get('keywords') or len(rules_data['keywords']) == 0:
            print("No valid keywords in rule, returning empty rule")
            return json.dumps({
                "label": "Empty Rule|No Valid Keywords",
                "keywords": []
            }, ensure_ascii=False, indent=2)
        
        # Check if keyword count is at least 2
        valid_keywords = [kw for kw in rules_data['keywords'] if isinstance(kw, dict) and kw.get('keyword') and str(kw.get('keyword')).strip()]
        if len(valid_keywords) < 1:
            print(f"Number of valid keywords in rule is less than 1 ({len(valid_keywords)}), returning empty rule")
            return json.dumps({
                "label": "Empty Rule|Insufficient Keyword Count",
                "keywords": []
            }, ensure_ascii=False, indent=2)
        
        # Re-serialize to JSON string
        return json.dumps(rules_data, ensure_ascii=False, indent=2)
        
    except json.JSONDecodeError as e:
        print(f"Rule JSON parsing failed: {str(e)}")
        print(f"Raw rule string: {rules_str[:200]}...")
        # Return empty rule instead of error rule
        return json.dumps({
            "label": "Empty Rule|Parsing Failed",
            "keywords": []
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Rule cleaning error: {str(e)}")
        return rules_str

async def process_single_device(device_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process single device data"""
    try:
        # Convert device data to JSON string
        res = json.dumps(device_data, indent=4)
        
        # Extract important data and new vocabulary
        important_data, new_word = await data_extract(res)
        
        # Prepare input for label agent
        if new_word != 'no':
            search_res = await search_agent(new_word)
            input_label_agent_one = f"{important_data}. Pay special attention to the explanation of keywords {search_res}"
        else:
            input_label_agent_one = important_data
        
        # Get primary label
        label_one = await label_one_agent(input_label_agent_one)
        
        # Parse primary label JSON
        label_one_json = None
        for i in range(MAX_RETRIES):
            try:
                print(f"[DEBUG] Attempting to parse primary label JSON: {repr(label_one)}")
                label_one_json = parse_json_from_ai_response(label_one)
                print(f"[DEBUG] Primary label JSON parsed successfully: {label_one_json}")
                break
            except (json.JSONDecodeError, ValueError) as e:
                print(f'Primary label JSON parsing failed, retrying {i+1}/{MAX_RETRIES}')
                print(f"[DEBUG] JSON parsing error: {str(e)}")
                print(f"[DEBUG] Error content: {repr(label_one[:200])}...")
                label_one = await label_one_agent(input_label_agent_one)
        
        if not label_one_json:
            raise ValueError("Cannot parse primary label JSON")
        
        big_class = label_one_json['main_category']
        reasoning = label_one_json['reasoning_process']
        
        # Simplified handling here, should implement async version of classify_query_simple in actual use
        retrieval = []  # await classify_query_simple_async(...)
        
        # Get subclass label
        if retrieval:
            subclass_input = f"{important_data}\n{reasoning}\nThe similar samples provided below are only for reference information and may not fully accurately reflect the target subclass, please treat them as auxiliary information rather than the final classification basis.\nEach sample contains two parts:\n- \"subclass\": Existing subclass label of the sample;\n- \"text\": Original information of the sample (such as IP, port, organization, geographic location, ASN, data content, etc.).\nPlease independently judge its subclass based on the characteristics of the target object, do not directly rely on the labels of similar samples.\n{json.dumps(retrieval, ensure_ascii=False, indent=4)}"
            label_subclass_res_str = await label_subclass(big_class, subclass_input)
        else:
            label_subclass_res_str = await label_subclass(big_class, f"{important_data}{reasoning}")
        
        # Parse subclass result
        try:
            label_subclass_res = parse_json_from_ai_response(label_subclass_res_str)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Subclass result JSON parsing failed: {str(e)}")
            print(f"[DEBUG] Error content: {repr(label_subclass_res_str[:200])}...")
            label_subclass_res = {"new_subclass_exists": "No", "subclass": "Other", "reasoning_process": "JSON parsing failed"}
        
        # Save new subclass
        if label_subclass_res['new_subclass_exists'] == 'Yes':
            try:
                file_name = f"{big_class}.txt"
                file_path = SUB_CLASS_DIR / file_name
                async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
                    await f.write(f"{label_subclass_res['subclass']}\n")
                print(f'New subclass {label_subclass_res["subclass"]} saved')
                
                # Clear cache to load latest data next time
                cache_key = get_cache_key(f"subclass_{file_name}")
                if (CACHE_DIR / f"{cache_key}.json").exists():
                    (CACHE_DIR / f"{cache_key}.json").unlink()
            except Exception as e:
                print(f"Error saving new subclass: {str(e)}")
        
        # Extract rules
        # Ensure category name is not empty
        big_class_name = big_class if big_class and big_class.strip() else "Unknown Main Category"
        subclass_name = label_subclass_res.get('subclass', 'Unknown Subclass')
        if not subclass_name or not subclass_name.strip():
            subclass_name = "Unknown Subclass"
        
        all_label = f"{big_class_name}|{subclass_name}"
        rules_raw = await rule_extract_agent(res, important_data, all_label)
        rules = clean_and_validate_rules(rules_raw)
        
        # Additional check: return empty rule if rule contains "JSON parsing failed"
        try:
            rules_data = json.loads(rules)
            if isinstance(rules_data.get('keywords'), list):
                for keyword_item in rules_data['keywords']:
                    if isinstance(keyword_item, dict) and keyword_item.get('keyword') == 'JSON parsing failed':
                        print("Detected rule contains 'JSON parsing failed', returning empty rule")
                        rules = json.dumps({
                            "label": "Empty Rule|Contains Invalid Keywords",
                            "keywords": []
                        }, ensure_ascii=False, indent=2)
                        break
        except json.JSONDecodeError:
            pass  # Keep as is if parsing fails
        
        return {
            "device_data": device_data,
            "important_data": important_data,
            "main_category": big_class,
            "subclass": label_subclass_res,
            "rules": rules
        }
        
    except Exception as e:
        print(f"Error processing device: {str(e)}")
        return {"error": str(e), "device_data": device_data}

async def process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process batch of device data"""
    tasks = [process_single_device(device) for device in batch]
    return await asyncio.gather(*tasks)

async def process_single_device(device_data: Dict[str, Any], device_index: int) -> Dict[str, Any]:
    """Process single device data, return only required fields"""
    try:
        # Convert device data to JSON string
        res = json.dumps(device_data, indent=4)
        print(f"[DEBUG] Processing device data {device_index}, JSON length: {len(res)}")
        print(f"[DEBUG] First 200 characters of device data: {repr(res[:200])}")
        
        # Extract important data and new vocabulary
        important_data, new_word = await data_extract(res)
        print(f"[DEBUG] Data extraction completed - important_data length: {len(important_data) if important_data else 0}")
        print(f"[DEBUG] Data extraction completed - new_word: {repr(new_word)}")
        
        # Prepare input for label agent
        if new_word != 'no':
            search_res = await search_agent(new_word)
            input_label_agent_one = f"{important_data}. Pay special attention to the explanation of keywords {search_res}"
        else:
            input_label_agent_one = important_data
        
        # Get primary label
        label_one = await label_one_agent(input_label_agent_one)
        
        # Parse primary label JSON
        label_one_json = None
        for i in range(MAX_RETRIES):
            try:
                print(f"[DEBUG] Attempting to parse primary label JSON: {repr(label_one)}")
                label_one_json = parse_json_from_ai_response(label_one)
                print(f"[DEBUG] Primary label JSON parsed successfully: {label_one_json}")
                break
            except (json.JSONDecodeError, ValueError) as e:
                print(f'Primary label JSON parsing failed, retrying {i+1}/{MAX_RETRIES}')
                print(f"[DEBUG] JSON parsing error: {str(e)}")
                print(f"[DEBUG] Error content: {repr(label_one[:200])}...")
                label_one = await label_one_agent(input_label_agent_one)
        
        if not label_one_json:
            raise ValueError("Cannot parse primary label JSON")
        
        big_class = label_one_json['main_category']
        reasoning = label_one_json['reasoning_process']
        
        # Simplified handling here, should implement async version of classify_query_simple in actual use
        retrieval = []  # await classify_query_simple_async(...)
        
        # Get subclass label
        if retrieval:
            subclass_input = f"{important_data}\n{reasoning}\nThe similar samples provided below are only for reference information and may not fully accurately reflect the target subclass, please treat them as auxiliary information rather than the final classification basis.\nEach sample contains two parts:\n- \"subclass\": Existing subclass label of the sample;\n- \"text\": Original information of the sample (such as IP, port, organization, geographic location, ASN, data content, etc.).\nPlease independently judge its subclass based on the characteristics of the target object, do not directly rely on the labels of similar samples.\n{json.dumps(retrieval, ensure_ascii=False, indent=4)}"
            label_subclass_res_str = await label_subclass(big_class, subclass_input)
        else:
            label_subclass_res_str = await label_subclass(big_class, f"{important_data}{reasoning}")
        
        # Parse subclass result
        try:
            label_subclass_res = parse_json_from_ai_response(label_subclass_res_str)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Subclass result JSON parsing failed: {str(e)}")
            print(f"[DEBUG] Error content: {repr(label_subclass_res_str[:200])}...")
            label_subclass_res = {"new_subclass_exists": "No", "subclass": "Other", "reasoning_process": "JSON parsing failed"}
        
        # Save new subclass
        if label_subclass_res['new_subclass_exists'] == 'Yes':
            try:
                file_name = f"{big_class}.txt"
                file_path = SUB_CLASS_DIR / file_name
                async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
                    await f.write(f"\n- {label_subclass_res['subclass']}")
                print(f'New subclass {label_subclass_res["subclass"]} saved')
                
                # Clear cache to load latest data next time
                cache_key = get_cache_key(f"subclass_{file_name}")
                if (CACHE_DIR / f"{cache_key}.json").exists():
                    (CACHE_DIR / f"{cache_key}.json").unlink()
            except Exception as e:
                print(f"Error saving new subclass: {str(e)}")
        
        # Extract rules
        # Process category name, remove content in parentheses
        processed_big_class = big_class
        if isinstance(processed_big_class, str):
            processed_big_class = re.sub(r'[\(（][^)\）]*[\)）]$', '', processed_big_class)
        
        processed_subclass = label_subclass_res.get('subclass', 'Unknown Subclass')
        if isinstance(processed_subclass, str):
            processed_subclass = re.sub(r'[\(（][^)\）]*[\)）]$', '', processed_subclass)
        
        # Ensure processed category name is not empty
        if not processed_big_class or not processed_big_class.strip():
            processed_big_class = "Unknown Main Category"
        if not processed_subclass or not processed_subclass.strip():
            processed_subclass = "Unknown Subclass"
        
        all_label = f"{processed_big_class}|{processed_subclass}"
        rules_raw = await rule_extract_agent(res, important_data, all_label)
        rules = clean_and_validate_rules(rules_raw)
        
        # Additional check: return empty rule if rule contains "JSON parsing failed"
        try:
            rules_data = json.loads(rules)
            if isinstance(rules_data.get('keywords'), list):
                for keyword_item in rules_data['keywords']:
                    if isinstance(keyword_item, dict) and keyword_item.get('keyword') == 'JSON parsing failed':
                        print("Detected rule contains 'JSON parsing failed', returning empty rule")
                        rules = json.dumps({
                            "label": "Empty Rule|Contains Invalid Keywords",
                            "keywords": []
                        }, ensure_ascii=False, indent=2)
                        break
        except json.JSONDecodeError:
            pass  # Keep as is if parsing fails
        
        # Return only final result and rules, include member_id and cluster_id from input data
        result = {
            "serial_number": device_index,
            "final_result": {
                "main_category": big_class,
                "subclass": label_subclass_res
            },
            "generated_rules": rules
        }
            
        return result
        

        
    except Exception as e:
        print(f"Error processing device: {str(e)}")
        result = {
            "index": device_index,
            "member_id": device_data.get("member_id"),
            "cluster_id": device_data.get("cluster_id"),
            "error": str(e)
        }
            
        return result


async def process_single_device_with_fallback(device_data: Dict[str, Any], device_index: int) -> Dict[str, Any]:
    """Wrapper function for processing single device data, ensuring a result is always returned"""
    try:
        return await process_single_device(device_data, device_index)
    except Exception as e:
        print(f"Exception occurred while processing device {device_index}: {str(e)}")
        # Return result containing error information, ensuring index matches
        return {
            "index": device_index,
            "member_id": device_data.get("member_id"),
            "cluster_id": device_data.get("cluster_id"),
            "error": f"Processing exception: {str(e)}"
        }

async def process_single_device_with_fallback_and_save(device_data: Dict[str, Any], device_index: int) -> Dict[str, Any]:
    """Wrapper function for processing single device data (does not save to file - saving logic is handled uniformly in process_batch)"""
    try:
        return await process_single_device(device_data, device_index)
    except Exception as e:
        print(f"Exception occurred while processing device {device_index}: {str(e)}")
        # Return result containing error information, ensuring index matches
        return {
            "index": device_index,
            "member_id": device_data.get("member_id"),
            "cluster_id": device_data.get("cluster_id"),
            "error": f"Processing exception: {str(e)}"
        }

async def process_batch(batch: List[Dict[str, Any]], start_index: int) -> Tuple[List[Dict[str, Any]], bool, List[Dict[str, Any]]]:
    """Process a batch of device data, ensuring each original record has a corresponding result (optimized version: batch write to file)"""
    results = []
    unmatched_devices = []  # Store devices and their indices that don't match any rules
    rules_path = "./rules.json"
    # Collect final_labels for all devices, sorted by device index, then write in batch at last
    final_labels: Dict[int, str] = {}  # {device_index: final_label}
    
    # Step 1: Read rules file once at batch level (shared reading to avoid repeated I/O)
    try:
        rules = load_rules_with_lock(rules_path, compile_for_performance=True)
        print(f"Batch started: Loaded {len(rules)} rules")
        rules_modified = False  # Flag to indicate if rules have been modified
    except Exception as e:
        print(f"Failed to load rules file: {str(e)}, will use per-read mode")
        rules = None
        rules_modified = False
    
    # Step 2: Check rule matching status for all devices (using shared rule list)
    for i, device in enumerate(batch):
        device_index = start_index + i
        
        # First call clean1 for rule matching (use verbose=False to improve performance)
        try:
            is_matched, updated_rules, final_label = clean1(device, rules_path, rules, verbose=False)
            if is_matched:
                print(f"Device {device_index+1} matched rules, skipping processing")
                # Save matched label
                if final_label:
                    final_labels[device_index] = final_label
                # Create skip result
                skip_result = {
                    "index": device_index,
                    "status": "skipped",
                    "reason": "Matched rules, skipped processing"
                }
                results.append(skip_result)
                # If updated rule list is returned, it means rules were modified
                if updated_rules is not None:
                    rules = updated_rules
                    rules_modified = True
            else:
                print(f"Device {device_index+1} did not match any rules, added to async processing queue")
                unmatched_devices.append((device, device_index))
        except Exception as e:
            print(f"Error matching rules for device {device_index+1}: {str(e)}, added to async processing queue")
            unmatched_devices.append((device, device_index))
    
    # Step 3: Asynchronously process all unmatched devices concurrently
    if unmatched_devices:
        print(f"Starting async processing for {len(unmatched_devices)} devices that didn't match rules")
        
        # Create async task list
        tasks = []
        for device, device_index in unmatched_devices:
            task = process_single_device_with_fallback_and_save(device, device_index)
            tasks.append(task)
        
        # Execute all tasks concurrently
        unmatched_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and collect final_labels (in original order)
        for i, result in enumerate(unmatched_results):
            device_index = unmatched_devices[i][1]
            
            if isinstance(result, Exception):
                # Handle exception cases
                error_result = {
                    "index": device_index,
                    "member_id": unmatched_devices[i][0].get("member_id"),
                    "cluster_id": unmatched_devices[i][0].get("cluster_id"),
                    "error": f"Processing exception: {str(result)}"
                }
                results.append(error_result)
                final_labels[device_index] = "Processing exception"
            else:
                # Normal processing results
                results.append(result)
                
                # Collect results for data that didn't match rules
                if "final_result" in result and "error" not in result:
                    main_category = result["final_result"]["main_category"]
                    subcategory_info = result["final_result"].get("subclass")
                    
                    # Process main category, remove content in parentheses
                    if isinstance(main_category, str):
                        main_category = re.sub(r'[\(\(][^\)\)]*[\)\)]$', '', main_category)
                    
                    if isinstance(subcategory_info, dict) and "subclass" in subcategory_info:
                        subcategory = subcategory_info["subclass"]
                        # Process subcategory, remove content in parentheses
                        if isinstance(subcategory, str):
                            subcategory = re.sub(r'[\(\(][^\)\)]*[\)\)]$', '', subcategory)
                        final_label = f"{main_category}|{subcategory}"
                    else:
                        final_label = f"{main_category}|Unknown"
                    
                    final_labels[device_index] = final_label
                elif "error" in result:
                    # Handle error cases
                    final_labels[device_index] = "Processing error"
    
    # Step 4: Batch write to final_label.txt (sorted by device index)
    final_label_path = "./final_label.txt"
    if final_labels:
        try:
            with open(final_label_path, "a", encoding="utf-8") as f_label:
                # Write in sorted order of device index
                for idx in sorted(final_labels.keys()):
                    f_label.write(final_labels[idx] + "\n")
            print(f"Batch completed: Batch wrote {len(final_labels)} labels to final_label.txt")
        except Exception as e:
            print(f"Failed to batch write to final_label.txt: {str(e)}")
    
    # Step 5: Do not save rules here, save uniformly at the end of batch processing (avoid repeated saving)
    # Return rule modification status, let caller decide whether to save
    return results, rules_modified, rules

def get_samples_cache_file_path(data_file_path: str) -> str:
    """Generate sample cache file path"""
    import hashlib
    file_hash = hashlib.md5(data_file_path.encode('utf-8')).hexdigest()[:8]
    cache_file = f"./samples_cache_{file_hash}.json"
    return cache_file

def load_random_samples(data_file_path: str, sample_size: int = 20000) -> List[Dict[str, Any]]:
    """Quickly randomly extract specified number of samples from data file (simplified version)
    
    Args:
        data_file_path: Path to data file
        sample_size: Target number of samples (default: 20000)
    """
    global _samples_cache, _samples_cache_file
    
    # Check memory cache
    if _samples_cache is not None and _samples_cache_file == data_file_path:
        print(f"✓ Using sample data from memory cache ({len(_samples_cache)} samples)")
        return _samples_cache
    
    # Check persistent cache file
    cache_file_path = get_samples_cache_file_path(data_file_path)
    if os.path.exists(cache_file_path):
        try:
            print(f"✓ Found persistent cache file, loading...")
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                if isinstance(cached_data, list) and len(cached_data) == sample_size:
                    _samples_cache = cached_data
                    _samples_cache_file = data_file_path
                    print(f"✓ Successfully loaded {len(cached_data)} samples, skipping re-sampling")
                    return _samples_cache
                else:
                    print(f"Warning: Cache file format mismatch, will re-sample")
        except Exception as e:
            print(f"Warning: Failed to load cache file: {e}, will re-sample")
    
    if not data_file_path or not os.path.exists(data_file_path):
        print(f"Warning: Data file does not exist: {data_file_path}, cannot perform rule validation")
        return []
    
    try:
        print(f"Starting quick random sampling (target sample count: {sample_size})...")
        samples = []
        
        # Detect file format and read samples
        with open(data_file_path, 'r', encoding='utf-8') as f:
            # Try reading first few lines to determine format
            preview_lines = []
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                preview_lines.append(line.strip())
            
            # Determine if it's JSONL format
            is_jsonl = False
            if len(preview_lines) >= 2:
                valid_count = 0
                for line in preview_lines[:5]:
                    if line:
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                valid_count += 1
                        except:
                            break
                if valid_count >= 2:
                    is_jsonl = True
            
            # Reopen file for reading
            f.seek(0)
            
            if is_jsonl:
                # JSONL format: read line by line
                lines = f.readlines()
                import random
                if len(lines) > sample_size:
                    # Randomly select lines
                    selected_indices = random.sample(range(len(lines)), sample_size)
                    for idx in sorted(selected_indices):
                        try:
                            obj = json.loads(lines[idx].strip())
                            if isinstance(obj, dict):
                                samples.append(obj)
                        except:
                            continue
                else:
                    # Insufficient lines in file, read all
                    for line in lines:
                        line = line.strip()
                        if line:
                            try:
                                obj = json.loads(line)
                                if isinstance(obj, dict):
                                    samples.append(obj)
                            except:
                                continue
            else:
                # JSON array format: need complete parsing
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        import random
                        if len(data) > sample_size:
                            samples = random.sample(data, sample_size)
                        else:
                            samples = data
                    elif isinstance(data, dict) and 'result' in data and 'hits' in data['result']:
                        # load-balancer format
                        hits = data['result']['hits']
                        import random
                        if len(hits) > sample_size:
                            samples = random.sample(hits, sample_size)
                        else:
                            samples = hits
                except json.JSONDecodeError:
                    print(f"Warning: Cannot parse JSON file: {data_file_path}")
                    return []
        
        # Save to cache
        if samples:
            _samples_cache = samples
            _samples_cache_file = data_file_path
            try:
                with open(cache_file_path, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, ensure_ascii=False, indent=2)
                print(f"✓ Saved {len(samples)} samples to cache file")
            except Exception as e:
                print(f"Warning: Failed to save cache file: {e}")
        
        print(f"✓ Sampling completed, obtained {len(samples)} samples in total")
        return samples
        
    except Exception as e:
        print(f"Error during sampling: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def rule_matches_device(rule: Dict[str, Any], device: Dict[str, Any], device_id: str = None) -> bool:
    """Check if rule matches device data (string matching with cache optimization)
    
    A match is considered successful if all keywords of the rule exist in the device data (after conversion to string)
    """
    global _device_string_cache
    
    # Get keyword list of the rule (support both 'Keywords' and 'keywords' for compatibility)
    keywords = rule.get('Keywords', rule.get('keywords', []))
    if not keywords:
        return False
    
    # Extract keyword strings
    keyword_strings = []
    for keyword_item in keywords:
        if isinstance(keyword_item, dict):
            keyword = keyword_item.get('keyword', '')
            if keyword and isinstance(keyword, str):
                keyword_strings.append(keyword)
        elif isinstance(keyword_item, str):
            keyword_strings.append(keyword_item)
    
    if not keyword_strings:
        return False
    
    # Try to use cached device string
    if device_id and device_id in _device_string_cache:
        device_str = _device_string_cache[device_id]
    else:
        # Convert device data to string (recursively process nested structures, optimized version)
        def dict_to_string(obj, max_depth=10):
            """Recursively convert dictionary to string (optimized version: limit depth and size)"""
            if max_depth <= 0:
                return str(obj)
            if isinstance(obj, dict):
                # Only process values, ignore keys (since keys are usually field names and unlikely to contain keywords)
                parts = []
                for v in obj.values():
                    if len(parts) > 1000:  # Limit string length to avoid excessive size
                        break
                    parts.append(dict_to_string(v, max_depth-1))
                return ' '.join(parts)
            elif isinstance(obj, list):
                parts = []
                for item in obj[:100]:  # Limit list length
                    if len(parts) > 1000:
                        break
                    parts.append(dict_to_string(item, max_depth-1))
                return ' '.join(parts)
            else:
                return str(obj)
        
        device_str = dict_to_string(device)
        
        # Cache device string (if device_id is provided)
        if device_id:
            _device_string_cache[device_id] = device_str
            # Limit cache size to avoid memory overflow
            if len(_device_string_cache) > 20000:
                # Delete oldest 50% of cache
                keys_to_remove = list(_device_string_cache.keys())[:10000]
                for key in keys_to_remove:
                    del _device_string_cache[key]
    
    # Check if all keywords exist in device string
    for keyword in keyword_strings:
        if keyword not in device_str:
            return False
    
    return True

def validate_rule_with_samples(rule: Dict[str, Any], samples: List[Dict[str, Any]], threshold: float = None) -> bool:
    """Validate if rule passes sample matching test (optimized version: early termination)
    
    Args:
        rule: Rule to validate
        samples: Randomly extracted sample data
        threshold: Matching ratio threshold, discard rule if exceeding this ratio (default: uses SAMPLE_VALIDATION_THRESHOLD global constant)
    
    Returns:
        True means rule passed validation (matching ratio <= threshold), False means discard rule (matching ratio > threshold)
    """
    # Use global constant if not specified
    if threshold is None:
        threshold = SAMPLE_VALIDATION_THRESHOLD
    
    if not samples:
        print("Warning: No sample data, skipping rule validation")
        return True  # Default to pass if no samples
    
    match_count = 0
    max_allowed_matches = int(len(samples) * threshold) + 1  # Maximum allowed matches (fail if exceeded)
    
    # Generate unique ID for each device for caching
    for idx, device in enumerate(samples):
        # Use index as device_id (better if device has unique identifier)
        device_id = f"device_{idx}"
        
        if rule_matches_device(rule, device, device_id=device_id):
            match_count += 1
            # Early termination: return False immediately if match count exceeds threshold
            if match_count > max_allowed_matches:
                match_ratio = match_count / len(samples)
                print(f"Rule matching test (early termination): {match_count}/{len(samples)} = {match_ratio:.4f} ({match_ratio*100:.2f}%)")
                print(f"Rule matching ratio {match_ratio*100:.2f}% exceeds threshold {threshold*100:.2f}%, discarding the rule")
                return False
    
    match_ratio = match_count / len(samples)
    
    print(f"Rule matching test: {match_count}/{len(samples)} = {match_ratio:.4f} ({match_ratio*100:.2f}%)")
    
    if match_ratio > threshold:
        print(f"Rule matching ratio {match_ratio*100:.2f}% exceeds threshold {threshold*100:.2f}%, discarding the rule")
        return False
    else:
        print(f"Rule matching ratio {match_ratio*100:.2f}% does not exceed threshold {threshold*100:.2f}%, keeping the rule")
        return True

def get_current_rules_count(rules_file_path: str = "./rules.json") -> int:
    """Get number of rules in current rules file"""
    try:
        if not os.path.exists(rules_file_path):
            return 0
        
        with open(rules_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return 0
            
            rules = json.loads(content)
            if isinstance(rules, list):
                return len(rules)
            else:
                return 0
    except Exception as e:
        print(f"Failed to get rule count: {e}")
        return 0

async def extract_and_append_rules(results: List[Dict[str, Any]], rules_file_path: str = "./rules.json", data_file_path: str = None, res_file_path: str = None, processed_count: int = 0, batch_num: int = 0, confidence_threshold: float = 0.6, enable_sample_validation: bool = True, sample_validation_threshold: float = None):
    """Extract new rules from processing results and save them, perform sample validation first, then use LLM to evaluate rule generality, and finally perform inverted index check
    
    Args:
        confidence_threshold: Confidence threshold for LLM evaluation, filter rules only when confidence >= this threshold (default: 0.6)
        enable_sample_validation: Whether to enable sample validation (default: True, perform sample validation first for quick filtering)
        sample_validation_threshold: Matching ratio threshold for sample validation, filter rules if exceeding this ratio (default: uses SAMPLE_VALIDATION_THRESHOLD global constant)
    """
    # Use global constant if not specified
    if sample_validation_threshold is None:
        sample_validation_threshold = SAMPLE_VALIDATION_THRESHOLD
    try:
        import json
        # Import functions from clean_rules_2 module
        from step.clean_rules_2 import update_rules, keep_rules_from_file, clean_rules
        
        # Directory to save deleted rules
        deleted_rules_dir = "./deleted_rules"
        os.makedirs(deleted_rules_dir, exist_ok=True)
        
        # Filter results containing rules
        valid_results = []
        for result in results:
            if (isinstance(result, dict) and 
                "final_result" in result and 
                "generated_rules" in result and 
                "error" not in result):
                valid_results.append(result)
        
        if not valid_results:
            print("No valid results containing rules found, skipping rule extraction")
            return 0
        
        print(f"\n{'='*60}")
        print(f"[Rule Extraction Phase] Found {len(valid_results)} valid results containing rules, starting rule extraction...")
        print(f"{'='*60}")
        
        # Create temporary file to save current batch results
        # DeviceClassificationAnalyzer expects Chinese field names: "最终结果" and "生成的规则".
        # Normalize our internal result format ("final_result"/"generated_rules") to that schema.
        normalized_results = []
        for r in valid_results:
            try:
                final_result = r.get("final_result")
                generated_rules = r.get("generated_rules")

                if isinstance(generated_rules, str):
                    try:
                        generated_rules_obj = parse_json_from_ai_response(generated_rules)
                    except Exception:
                        generated_rules_obj = None
                elif isinstance(generated_rules, dict):
                    generated_rules_obj = generated_rules
                else:
                    generated_rules_obj = None

                # Extract label for analyzer: prefer rule.label, else build from final_result
                label = None
                if isinstance(generated_rules_obj, dict):
                    label = generated_rules_obj.get("label")
                if not label and isinstance(final_result, dict):
                    main_category = final_result.get("main_category")
                    sub_info = final_result.get("subclass")
                    sub_name = None
                    if isinstance(sub_info, dict):
                        sub_name = sub_info.get("subclass")
                    elif isinstance(sub_info, str):
                        sub_name = sub_info
                    if main_category and sub_name:
                        label = f"{main_category}|{sub_name}"

                # Convert keywords schema to analyzer's expected "关键词" list
                keywords_cn = []
                if isinstance(generated_rules_obj, dict):
                    kws = generated_rules_obj.get("keywords")
                    if isinstance(kws, list):
                        for kw in kws:
                            if isinstance(kw, dict):
                                val = kw.get("keyword")
                                if isinstance(val, str) and val.strip():
                                    keywords_cn.append({"关键词": val.strip()})
                            elif isinstance(kw, str) and kw.strip():
                                keywords_cn.append({"关键词": kw.strip()})

                rule_cn_obj = {"关键词": keywords_cn}
                normalized_results.append({
                    "最终结果": {"类别": label or "Unknown Category|Unknown Subcategory"},
                    "生成的规则": json.dumps(rule_cn_obj, ensure_ascii=False)
                })
            except Exception:
                # Skip malformed entry conservatively
                continue

        temp_file = "./temp_results.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(normalized_results, f, ensure_ascii=False, indent=2)
        
        # Extract rules using DeviceClassificationAnalyzer
        analyzer = DeviceClassificationAnalyzer(temp_file, auto_fix=True)
        flat_rules = analyzer.extract_flat_rules()
        
        # Filter out rules with empty keywords and rules containing "JSON parsing failed"
        # Also filter out rules with fewer than 2 keywords (check before generalization)
        filtered_rules = []
        for rule in flat_rules:
            # DeviceClassificationAnalyzer returns keys: 'Label' and 'Keywords'
            if not rule.get('Keywords'):
                continue
            
            # Check keyword count (before generalization)
            keywords = rule.get('Keywords', [])
            if isinstance(keywords, list):
                keyword_count = len([kw for kw in keywords if kw and str(kw).strip()])
            else:
                keyword_count = 1 if keywords else 0
            
            if keyword_count < 1:
                print(f"Filtering out rule with fewer than 2 keywords (before generalization): {rule.get('label', 'Unknown label')}, keyword count: {keyword_count}")
                continue
            
            # Check if contains "JSON parsing failed" keyword
            has_json_parse_failure = False
            if isinstance(rule.get('keywords'), list):
                for keyword_item in rule['keywords']:
                    if isinstance(keyword_item, dict) and keyword_item.get('keyword') == 'JSON parsing failed':
                        has_json_parse_failure = True
                        break
            
            if not has_json_parse_failure:
                filtered_rules.append(rule)
            else:
                print(f"Filtering out rule containing 'JSON parsing failed': {rule.get('label', 'Unknown label')}")
        
        if not filtered_rules:
            print("No valid rules extracted, skipping save")
            return 0
        
        print(f"Extracted {len(filtered_rules)} valid rules")
        
        # Filter rules: labels containing "Other", keywords containing "UNKNOWN" or "member", containing "JSON parsing failed", keyword count <=2
        # Then use LLM to evaluate if rules are too generic
        rules_to_evaluate = []  # Rules to be evaluated
        
        print(f"Starting further rule filtering, total {len(filtered_rules)} rules")
        
        for rule in filtered_rules:
            # 1. Filter out rules with labels containing "Other"
            # DeviceClassificationAnalyzer returns 'Label' and 'Keywords' (capitalized)
            label = rule.get('Label', rule.get('label', ''))
            if isinstance(label, str) and 'Other' in label:
                print(f"Filtering out rule with label containing 'Other': {label}")
                continue
            
            # Check both 'Keywords' (capitalized) and 'keywords' (lowercase) for compatibility
            keywords = rule.get('Keywords', rule.get('keywords', []))
            if not keywords:
                continue
            
            # 2. Basic filtering: check if keywords contain invalid keywords
            has_invalid_keyword = False
            keyword_count = 0
            generic_keyword_count = 0  # Count generic keyword quantity
            is_service_device_label = 'Service Device|Application Service Device' in label  # Whether it's service device label
            
            # Define generic keyword list (specifically for service device labels)
            generic_keywords_list = ['http', 'https', 'web', 'server', '200 ok', '302 found', '401 unauthorized', 
                                   '404 not found', 'content-type', 'content-length', 'server', 'date', 
                                   'connection', 'cache-control', 'sameorigin', 'x-frame-options']
            
            if isinstance(keywords, list):
                for keyword_item in keywords:
                    keyword_count += 1
                    # Get string value of keyword
                    keyword_str = ''
                    if isinstance(keyword_item, dict):
                        keyword_str = str(keyword_item.get('keyword', ''))
                    elif isinstance(keyword_item, str):
                        keyword_str = keyword_item
                    else:
                        keyword_str = str(keyword_item)
                    
                    keyword_lower = keyword_str.lower().strip()
                    
                    # Check if contains "UNKNOWN" or "member"
                    keyword_upper = keyword_str.upper()
                    if 'UNKNOWN' in keyword_upper or 'MEMBER' in keyword_upper:
                        has_invalid_keyword = True
                        print(f"Filtering out rule with invalid keywords: {label} (keyword: {keyword_str})")
                        break
                    
                    # Check if contains "JSON parsing failed"
                    if 'JSON parsing failed' in keyword_str:
                        has_invalid_keyword = True
                        print(f"Filtering out rule containing 'JSON parsing failed': {label}")
                        break
                    
                    # For "Service Device|Application Service Device" labels, check if contains overly generic keywords
                    if is_service_device_label:
                        if keyword_lower in generic_keywords_list or keyword_lower in ['yes', 'no', 'true', 'false']:
                            generic_keyword_count += 1
                            print(f"Warning: Service device label rule contains generic keyword: {keyword_str}")
            
            if has_invalid_keyword:
                continue
            
            # # 3. Filter out rules with keyword count <=2
            # if keyword_count < 2:
            #     print(f"Filtering out rule with keyword count <=2: {label} (keyword count: {keyword_count})")
            #     continue
            
            # 4. For "Service Device|Application Service Device" labels, filter directly if all keywords are generic
            if is_service_device_label and generic_keyword_count > 0 and generic_keyword_count >= keyword_count:
                print(f"Filtering out service device label rule (all keywords are generic): {label} (generic keywords: {generic_keyword_count}/{keyword_count})")
                continue
            
            # 5. Rules passing basic filtering first undergo sample validation (if enabled)
            if enable_sample_validation and data_file_path:
                # Sample validation will be performed uniformly later, add to pending validation list first
                rules_to_evaluate.append(rule)
            else:
                # If sample validation not enabled, add directly to LLM evaluation list
                rules_to_evaluate.append(rule)
        
        # 6. Perform sample validation first (if enabled) to quickly filter obviously overly broad rules
        sample_validated_rules = []
        if enable_sample_validation and data_file_path:
            print(f"\nStep 1: Sample Validation (quick filtering of overly broad rules)...")
            print(f"  Data file: {data_file_path}")
            print(f"  Matching ratio threshold: {sample_validation_threshold*100:.1f}%")
            
            samples = load_random_samples(data_file_path, sample_size=20000)
            
            if samples:
                print(f"  Loaded {len(samples)} samples for validation")
                
                for i, rule in enumerate(rules_to_evaluate, 1):
                    if validate_rule_with_samples(rule, samples, threshold=sample_validation_threshold):
                        sample_validated_rules.append(rule)
                    else:
                        rule_label = rule.get('Label', rule.get('label', 'Unknown label'))
                        print(f"  Sample validation filtered: {rule_label} (matching ratio exceeded threshold)")
                
                print(f"  Sample validation completed: {len(sample_validated_rules)}/{len(rules_to_evaluate)} rules passed validation")
                if len(sample_validated_rules) < len(rules_to_evaluate):
                    print(f"  Sample validation filtered out {len(rules_to_evaluate) - len(sample_validated_rules)} rules")
                
                # Use rules passing sample validation for LLM evaluation
                rules_to_evaluate = sample_validated_rules
            else:
                print("  Warning: Cannot load sample data, skipping sample validation, all rules will proceed to LLM evaluation")
        
        if not rules_to_evaluate:
            print("No rules passed sample validation, skipping save")
            return 0
        
        # 7. Use LLM to batch evaluate if rules are too generic (only evaluate rules passing sample validation)
        print(f"\nStep 2: LLM evaluation of generality for {len(rules_to_evaluate)} rules...")
        
        # Create evaluation tasks
        evaluation_tasks = [evaluate_rule_genericness(rule) for rule in rules_to_evaluate]
        
        # Execute evaluation concurrently (limit concurrency to avoid API rate limiting)
        batch_size = 5  # Evaluate 5 rules per batch
        llm_validated_rules = []
        for i in range(0, len(evaluation_tasks), batch_size):
            batch_tasks = evaluation_tasks[i:i+batch_size]
            batch_rules = rules_to_evaluate[i:i+batch_size]
            
            try:
                # Evaluate current batch concurrently
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process evaluation results
                for rule, evaluation in zip(batch_rules, batch_results):
                    label = rule.get('Label', rule.get('label', 'Unknown label'))
                    
                    if isinstance(evaluation, Exception):
                        print(f"Error evaluating rule: {str(evaluation)}, conservative approach: keep the rule")
                        print(f"  Rule: {label} (confidence: cannot evaluate)")
                        # Conservative approach when error occurs: keep the rule
                        llm_validated_rules.append(rule)
                        continue
                    
                    confidence = evaluation.get("confidence", 0.0)
                    is_too_generic = evaluation.get("is_too_generic", False)
                    reason = evaluation.get("reason", "")
                    generic_keywords = evaluation.get("generic_keywords", [])
                    
                    # Output confidence for each rule
                    if is_too_generic:
                        # Filter only when confidence exceeds threshold
                        if confidence >= confidence_threshold:
                            print(f"❌ Filtering rule: {label} (confidence: {confidence:.2f}, threshold: {confidence_threshold:.2f}) - determined to be too generic")
                            if generic_keywords:
                                print(f"   Generic keywords: {', '.join(generic_keywords[:3])}...")  # Show only first 3
                            if reason:
                                print(f"   Reason: {reason[:100]}...")  # Show only first 100 characters
                            continue
                        else:
                            # Confidence below threshold, keep the rule
                            print(f"✓ Keeping rule: {label} (confidence: {confidence:.2f} < {confidence_threshold:.2f}) - insufficient confidence, keeping")
                            llm_validated_rules.append(rule)
                            continue
                    else:
                        # Rule not determined to be too generic, keep it
                        print(f"✓ Keeping rule: {label} (confidence: {confidence:.2f}) - determined to be valid rule")
                        llm_validated_rules.append(rule)
            except Exception as e:
                print(f"Error in batch evaluation: {str(e)}, conservative approach: keep all rules in current batch")
                # Conservative approach when error occurs: keep all rules in current batch
                llm_validated_rules.extend(batch_rules)
            
            # Show progress
            if (i + batch_size) % 20 == 0 or (i + batch_size) >= len(evaluation_tasks):
                print(f"  Evaluated {min(i + batch_size, len(evaluation_tasks))}/{len(evaluation_tasks)} rules...")
        
        if not llm_validated_rules:
            print("No rules passed LLM evaluation, skipping save")
            return 0
        
        print(f"Extracted {len(llm_validated_rules)} valid rules (passed sample validation, LLM evaluation and inverted index check)")
        
        # Phase 1: Collect all rule deletion information (do not actually delete)
        all_remove_info = []
        kept_rules = []  # Collect new rules that passed validation
        
        # First read current rules file for subsequent clean_rules calls
        try:
            with open(rules_file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    current_rules = []
                else:
                    current_rules = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            current_rules = []
        
        print(f"Current rules file contains {len(current_rules)} rules")
        print(f"Step 3: Inverted Index Check (retain original clean_rules check)...")
        print(f"Starting to process {len(llm_validated_rules)} new rules one by one, each rule needs to perform similarity matching with {len(current_rules)} existing rules...")
       
        for i, new_rule in enumerate(llm_validated_rules):
            print(f"\n{'='*60}")
            print(f"[Progress] Processing {i+1}/{len(filtered_rules)} new rule")
            
            # Convert rule format from 'Label'/'Keywords' to 'label'/'keywords' for clean_rules/update_rules compatibility
            normalized_rule = {
                'label': new_rule.get('Label', new_rule.get('label', 'Unknown')),
                'keywords': new_rule.get('Keywords', new_rule.get('keywords', []))
            }
            
            print(f"Rule label: {normalized_rule.get('label', 'Unknown')}")
            print(f"Rule keywords: {normalized_rule.get('keywords', [])}")
            print(f"Starting to call update_rules for validation (will match with {len(current_rules)} existing rules)...")
            
            # Get matches before calling update_rules for subsequent index conversion
            # update_rules reads rules file internally, so we need to get matches before calling
            threshold = 0.15
            matches = clean_rules(normalized_rule, threshold, rules=current_rules, rules_file=rules_file_path)
            
            # Call update_rules for validation and collect deletion information
            keep, rm = update_rules(rules_file_path, data_file_path, res_file_path, normalized_rule, processed_count)
            
            print(f"Rule {i+1} processing completed: keep={keep}, number of rules to delete={len(rm) if rm else 0}")
            

            # rm=[]   ####Set this to empty
            # keep=True   ###Set this to True for ablation experiment

            # update_rules may modify rules file, so re-read current_rules
            try:
                with open(rules_file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        current_rules = []
                    else:
                        current_rules = json.loads(content)
            except (json.JSONDecodeError, FileNotFoundError):
                current_rules = []

            # Determine if new rule is kept based on keep value
            if keep:
                print(f"Rule {i+1} passed validation and will be kept")
                # Normalize rule format to lowercase for consistency with existing rules
                normalized_kept_rule = {
                    'label': normalized_rule.get('label', 'Unknown'),
                    'keywords': normalized_rule.get('keywords', [])
                }
                kept_rules.append(normalized_kept_rule)
                
                # Collect deletion information if there are old rules to delete
                if rm!=[]:
                    print(f"Found {len(rm)} overlapping rules to delete")
                    for rm_index in rm:
                        # rm_index is index in matches, need to convert to index in current_rules
                        if 0 <= rm_index < len(matches):
                            actual_rule_index = matches[rm_index].get('rule_index', rm_index)
                            # Check if index is valid
                            if 0 <= actual_rule_index < len(current_rules):
                                old_rule = current_rules[actual_rule_index]
                                all_remove_info.append({
                                    'rule_index': actual_rule_index,
                                    'reason': 'Overwritten by new rule',
                                    'label': old_rule.get('label', 'Unknown'),
                                    'keywords': old_rule.get('keywords', [])
                                })
                            else:
                                print(f"Warning: Rule index {actual_rule_index} out of range (current rule count: {len(current_rules)}), skipping")
                        else:
                            print(f"Warning: matches index {rm_index} out of range (matches length: {len(matches)}), skipping")
            else:
                print(f"Rule {i+1} failed validation and will be discarded")
                # Collect deletion information if there are old rules to delete
                if rm!=[]:
                    print(f"Also found {len(rm)} old rules to delete")
                    for rm_index in rm:
                        # rm_index is index in matches, need to convert to index in current_rules
                        if 0 <= rm_index < len(matches):
                            actual_rule_index = matches[rm_index].get('rule_index', rm_index)
                            # Check if index is valid
                            if 0 <= actual_rule_index < len(current_rules):
                                old_rule = current_rules[actual_rule_index]
                                all_remove_info.append({
                                    'rule_index': actual_rule_index,
                                    'reason': 'Discarded along with new rule',
                                    'label': old_rule.get('label', 'Unknown'),
                                    'keywords': old_rule.get('keywords', [])
                                })
                            else:
                                print(f"Warning: Rule index {actual_rule_index} out of range (current rule count: {len(current_rules)}), skipping")
                        else:
                            print(f"Warning: matches index {rm_index} out of range (matches length: {len(matches)}), skipping")
        
        # Phase 2: Perform deletion and addition operations uniformly
        print(f"\n=== Batch Processing Results ===")
        print(f"New rules passed validation: {len(kept_rules)}")
        print(f"Existing rules to delete: {len(all_remove_info)}")
        
        # Deduplicate rules to delete
        unique_remove_info = []
        seen_rule_indices = set()
        for info in all_remove_info:
            rule_index = info['rule_index']
            if rule_index not in seen_rule_indices:
                seen_rule_indices.add(rule_index)
                unique_remove_info.append(info)
        
        # Save deleted rules to separate file
        # Use actual number of deleted rules after deduplication
        deleted_rules_count = len(unique_remove_info) if unique_remove_info else 0
        # Count as deletion operation if there are rules to delete (even if empty after deduplication)
        if all_remove_info:
            deleted_rules_file = os.path.join(deleted_rules_dir, f"batch_{batch_num}_deleted_rules.json")
            deleted_rules_data = {
                "batch_num": batch_num,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_identified": len(all_remove_info),  # Total identified rules to delete
                "actually_deleted": len(unique_remove_info),  # Actually deleted rules
                "deleted_rules": []
            }
            
            # Collect detailed information of deleted rules (only save actually deleted ones)
            for info in unique_remove_info:
                rule_index = info['rule_index']
                if rule_index < len(current_rules):
                    deleted_rule = current_rules[rule_index].copy()
                    deleted_rule['deletion_info'] = {
                        'reason': info['reason'],
                        'batch_num': batch_num,
                        'deleted_at': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    deleted_rules_data['deleted_rules'].append(deleted_rule)
            
            # Save deleted rules file
            with open(deleted_rules_file, 'w', encoding='utf-8') as f:
                json.dump(deleted_rules_data, f, ensure_ascii=False, indent=2)
            
            print(f"This batch has deletion operations, identified {len(all_remove_info)} rules to delete in total")
            if len(unique_remove_info) < len(all_remove_info):
                print(f"Actually deleted {len(unique_remove_info)} rules after deduplication")
            print(f"Deleted rule information saved to: {deleted_rules_file}")
        
        # Perform deletion operation
        if unique_remove_info:
            print(f"\nStarting to delete {len(unique_remove_info)} conflicting rules...")
            
            # Get indices of rules to delete
            indices_to_remove = {info['rule_index'] for info in unique_remove_info}
            
            # Create new rule list, keep only rules not deleted
            final_rules = []
            for i, rule in enumerate(current_rules):
                if i not in indices_to_remove:
                    final_rules.append(rule)
            
            print(f"Retained {len(final_rules)} existing rules after deletion")
            for info in unique_remove_info:
                print(f"  Deleted rule {info['rule_index']}: label={info['label']}, keywords={info['keywords']}")
        else:
            print("\nNo rules to delete")
            final_rules = current_rules.copy()
        
        # Add kept rules to final_rules (update_rules may have already added them, but we ensure consistency)
        # Note: update_rules already adds rules to the file, but we add them here for clarity
        for kept_rule in kept_rules:
            # Check if rule already exists (avoid duplicates)
            rule_exists = False
            kept_label = kept_rule.get('label', '')
            kept_keywords = kept_rule.get('keywords', [])
            for existing_rule in final_rules:
                existing_label = existing_rule.get('label', existing_rule.get('Label', ''))
                existing_keywords = existing_rule.get('keywords', existing_rule.get('Keywords', []))
                if kept_label == existing_label and kept_keywords == existing_keywords:
                    rule_exists = True
                    break
            if not rule_exists:
                final_rules.append(kept_rule)
        
        # Write back to rules file
        with open(rules_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_rules, f, ensure_ascii=False, indent=2)
        
        print(f"\nBatch processing completed!")
        print(f"Final rules file contains {len(final_rules)} rules")
        print(f"  - Retained existing rules: {len(final_rules) - len(kept_rules)}")
        print(f"  - Newly added rules: {len(kept_rules)}")
        print(f"  - Deleted rules: {len(all_remove_info)} (actually deleted: {len(unique_remove_info)})")
        
        print(f"Batch rule processing completed, processed {len(llm_validated_rules)} new rules (passed sample validation, LLM evaluation and inverted index check)")
        
        # Clean up temporary files
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Return number of deleted rules for tracking
        return deleted_rules_count
            
    except Exception as e:
        print(f"Error extracting and appending rules: {e}")
        import traceback
        traceback.print_exc()
        return 0

async def detect_file_format(file_path: str) -> str:
    """Detect file format: 'json_array', 'jsonl', 'load_balancer', 'unknown'"""
    try:
        # Only read first 50KB to detect format
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            preview = await f.read(51200)
            
        preview = preview.strip()
        if not preview:
            return 'unknown'
        
        # Check if it's JSONL format (one JSON object per line)
        lines = preview.split('\n')
        valid_jsonl_count = 0
        if len(lines) >= 2:
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            valid_jsonl_count += 1
                    except:
                        break
            if valid_jsonl_count >= 2:
                return 'jsonl'
        
        # Check if it's JSON array format
        preview_clean = preview.lstrip()
        if preview_clean.startswith('['):
            return 'json_array'
        
        # Check if it's load-balancer format
        if preview_clean.startswith('{') and '"result"' in preview and '"hits"' in preview:
            return 'load_balancer'
        
        return 'unknown'
    except Exception as e:
        print(f"Error detecting file format: {e}")
        return 'unknown'

async def read_json_array_streaming(file_path: str):
    """Stream read JSON array, yield device data one by one"""
    device_index = 0
    decoder = json.JSONDecoder()
    
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
        # Read file but process in chunks
        buffer = ""
        skip_first_bracket = True
        
        async for chunk in file:
            buffer += chunk
            
            # Skip first '['
            if skip_first_bracket:
                pos = buffer.find('[')
                if pos != -1:
                    buffer = buffer[pos+1:].lstrip()
                    skip_first_bracket = False
                else:
                    continue
            
            # Try to parse complete JSON object
            while buffer.strip():
                buffer = buffer.lstrip()
                if not buffer or buffer.strip().startswith(']'):
                    break
                
                try:
                    # Try to parse one complete JSON object
                    obj, idx = decoder.raw_decode(buffer)
                    
                    if isinstance(obj, dict):
                        obj['member_id'] = f"member_{device_index+1}"
                        obj['cluster_id'] = f"cluster_{device_index+1}"
                        device_index += 1
                        yield obj
                    
                    # Remove parsed part
                    buffer = buffer[idx:].lstrip()
                    # Skip commas and whitespace
                    while buffer and buffer[0] in [',', '\n', '\r', ' ', '\t']:
                        buffer = buffer[1:]
                    
                    # Stop if closing bracket is encountered
                    if buffer.strip().startswith(']'):
                        break
                        
                except json.JSONDecodeError:
                    # Object may not be complete yet, need to continue reading
                    break

async def read_jsonl_streaming(file_path: str):
    """Stream read JSONL format, yield device data line by line"""
    device_index = 0
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
        async for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    obj['member_id'] = f"member_{device_index+1}"
                    obj['cluster_id'] = f"cluster_{device_index+1}"
                    device_index += 1
                    yield obj
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSONL line: {e}, content: {line[:100]}")

async def read_load_balancer_streaming(file_path: str):
    """Stream read load-balancer format (still need special handling if entire file is one object)"""
    # load-balancer format is usually a single large JSON object, hard to truly stream process
    # But we can still use chunk reading to reduce memory peak
    device_index = 0
    
    # Use incremental JSON parsing
    decoder = json.JSONDecoder()
    buffer = ""
    
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
        async for chunk in file:
            buffer += chunk
            try:
                # Try to parse complete JSON object
                obj, idx = decoder.raw_decode(buffer)
                
                if isinstance(obj, dict) and 'result' in obj and 'hits' in obj['result']:
                    hits = obj['result']['hits']
                    for device in hits:
                        device['member_id'] = f"member_{device_index+1}"
                        device['cluster_id'] = f"cluster_{device_index+1}"
                        device_index += 1
                        yield device
                    break
                else:
                    # If not load-balancer format, may need to continue reading
                    pass
            except json.JSONDecodeError:
                # JSON not complete yet, continue reading
                continue

async def process_devices(file_path: str, output_file: str) -> List[Dict[str, Any]]:
    """Process all device data in file, save immediately after each batch is completed"""
    try:
        # Detect file format and use streaming read (save memory, do not load entire file at once)
        file_format = await detect_file_format(file_path)
        print(f"Detected file format: {file_format}")
        
        # Select appropriate streaming read method based on format
        if file_format == 'jsonl':
            device_generator = read_jsonl_streaming(file_path)
            print("Using JSONL streaming read mode")
        elif file_format == 'load_balancer':
            device_generator = read_load_balancer_streaming(file_path)
            print("Using load-balancer streaming read mode")
        elif file_format == 'json_array':
            device_generator = read_json_array_streaming(file_path)
            print("Using JSON array streaming read mode")
        else:
            print("Cannot determine file format, trying JSON array streaming read")
            device_generator = read_json_array_streaming(file_path)
        
        # Process in batches (streaming read, do not load all data at once)
        results = []
        batch = []
        batch_num = 0
        total_processed_count = 0  # Cumulative total of actually processed data (excluding skipped, including LLM labeled and processing errors)
        device_count = 0  # Total number of all devices (including skipped and actually processed)
        
        print("Starting streaming read and processing of data...")
        
        async for device in device_generator:
            batch.append(device)
            device_count += 1
            
            # Process current batch when batch reaches BATCH_SIZE
            if len(batch) >= BATCH_SIZE:
                batch_num += 1
                start_index = device_count - len(batch)
                print(f"Processing batch {batch_num}, {len(batch)} devices in total (cumulative {device_count}th device)")
                
                # Process current batch
                batch_results, batch_rules_modified, batch_rules = await process_batch(batch, start_index)
                results.extend(batch_results)
                
                # Save immediately after each batch is processed
                print(f"Batch {batch_num} processing completed, saving results...")
                
                # Filter out skipped results, only save actually processed results
                filtered_results = [result for result in batch_results if result.get("status") != "skipped"]
                
                # Modify JSON serialization parameters to use less indentation
                json_kwargs = {
                    'ensure_ascii': False,
                    'indent': 2,  # Reduce indentation to 2 spaces
                    'separators': (', ', ': ')  # Optimize separators
                }
                
                # Only save to file if there are actually processed results
                if filtered_results:
                    # Write directly if it's the first batch
                    if batch_num == 1:
                        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                            await f.write(json.dumps(filtered_results, **json_kwargs))
                    else:
                        # For non-first batches, need to read existing data and append
                        try:
                            async with aiofiles.open(output_file, 'r', encoding='utf-8') as rf:
                                existing_content = await rf.read()
                            
                            # Parse existing data
                            existing_data = json.loads(existing_content) if existing_content.strip() else []
                            
                            # Append new data
                            existing_data.extend(filtered_results)
                            
                            # Write back complete content
                            async with aiofiles.open(output_file, 'w', encoding='utf-8') as wf:
                                await wf.write(json.dumps(existing_data, **json_kwargs))
                        except (json.JSONDecodeError, FileNotFoundError):
                            # Recreate if file is corrupted or does not exist
                            async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                                await f.write(json.dumps(filtered_results, **json_kwargs))
                
                skipped_count = len([r for r in batch_results if r.get("status") == "skipped"])
                batch_processed_count = len(filtered_results)
                print(f"Batch {batch_num} results saved - processed: {batch_processed_count}, skipped: {skipped_count}")
                
                # Check if batch contains data labeled by LLM
                llm_processed_count = 0
                for result in batch_results:
                    if (isinstance(result, dict) and 
                        "final_result" in result and 
                        "generated_rules" in result and 
                        "error" not in result and 
                        result.get("status") != "skipped"):
                        llm_processed_count += 1
                
                # Only update rules for batches with LLM labeled data
                deleted_rules_count = 0
                rules_file = "./rules.json"
                if llm_processed_count > 0:
                    # If rules were modified (downgrade processing), save rules file first to ensure extract_and_append_rules reads latest rules
                    if batch_rules_modified and batch_rules is not None:
                        try:
                            save_rules_with_lock(rules_file, batch_rules)
                            print(f"Batch {batch_num} preprocessing: saved downgraded rules file ({len(batch_rules)} rules)")
                        except Exception as e:
                            print(f"Failed to save rules file: {str(e)}")
                    
                    print(f"\n{'='*60}")
                    print(f"[Batch {batch_num} Rule Update Phase]")
                    print(f"{llm_processed_count} data entries in batch {batch_num} were labeled by LLM, starting new rule extraction...")
                    print(f"Note: Rule extraction and validation may take a long time, especially when there are a large number of rules")
                    print(f"{'='*60}")
                    # Use dynamic file path
                    final_label_file = "./final_label.txt"
                    # Use device_count as processed_count since it includes position of all processed data (including skipped)
                    deleted_rules_count = await extract_and_append_rules(filtered_results, rules_file, file_path, final_label_file, device_count, batch_num)
                    print(f"\nBatch {batch_num} rule update completed!")
                else:
                    print(f"All data in batch {batch_num} matched rules, no need to update rules file")
                    # If rules were modified (downgrade processing), need to save rules file
                    if batch_rules_modified and batch_rules is not None:
                        try:
                            save_rules_with_lock(rules_file, batch_rules)
                            print(f"Batch {batch_num} completed: saved updated rules file ({len(batch_rules)} rules)")
                        except Exception as e:
                            print(f"Failed to save rules file: {str(e)}")
                
                # Update cumulative total of processed data
                total_processed_count += batch_processed_count
                
                # Get current number of rules
                current_rules_count = get_current_rules_count("./rules.json")
                
                # Record batch processing status to rule growth tracker
                record_batch_processing(batch_num, total_processed_count, len(batch), current_rules_count, deleted_rules_count)
                
                # Clear batch for next iteration
                batch = []
        
        # Process remaining small batch if any
        if batch:
            batch_num += 1
            start_index = device_count - len(batch)
            print(f"Processing final batch {batch_num}, {len(batch)} devices in total (cumulative {device_count}th device)")
            
            batch_results, batch_rules_modified, batch_rules = await process_batch(batch, start_index)
            results.extend(batch_results)
            
            # Save final batch results
            filtered_results = [result for result in batch_results if result.get("status") != "skipped"]
            if filtered_results:
                json_kwargs = {
                    'ensure_ascii': False,
                    'indent': 2,
                    'separators': (', ', ': ')
                }
                try:
                    async with aiofiles.open(output_file, 'r', encoding='utf-8') as rf:
                        existing_content = await rf.read()
                    existing_data = json.loads(existing_content) if existing_content.strip() else []
                    existing_data.extend(filtered_results)
                    async with aiofiles.open(output_file, 'w', encoding='utf-8') as wf:
                        await wf.write(json.dumps(existing_data, **json_kwargs))
                except (json.JSONDecodeError, FileNotFoundError):
                    async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(filtered_results, **json_kwargs))
            
            skipped_count = len([r for r in batch_results if r.get("status") == "skipped"])
            batch_processed_count = len(filtered_results)
            print(f"Final batch {batch_num} results saved - processed: {batch_processed_count}, skipped: {skipped_count}")
            
            # Check if final batch contains data labeled by LLM
            llm_processed_count = 0
            for result in batch_results:
                if (isinstance(result, dict) and 
                    "final_result" in result and 
                    "generated_rules" in result and 
                    "error" not in result and 
                    result.get("status") != "skipped"):
                    llm_processed_count += 1
            
            # Only update rules for batches with LLM labeled data
            deleted_rules_count = 0
            rules_file = "./rules.json"
            if llm_processed_count > 0:
                # If rules were modified (downgrade processing), save rules file first to ensure extract_and_append_rules reads latest rules
                if batch_rules_modified and batch_rules is not None:
                    try:
                        save_rules_with_lock(rules_file, batch_rules)
                    except Exception as e:
                        print(f"{str(e)}")
                
                print(f"\n{'='*60}")
                print(f"[Final Batch {batch_num} Rule Update Phase]")
                print(f"{llm_processed_count} data entries in final batch {batch_num} were labeled by LLM, starting new rule extraction...")
                print(f"{'='*60}")
                
                final_label_file = "./final_label.txt"
                deleted_rules_count = await extract_and_append_rules(filtered_results, rules_file, file_path, final_label_file, device_count, batch_num)
                print(f"\nFinal batch {batch_num} rule update completed!")
                
            else:
                if batch_rules_modified and batch_rules is not None:
                    try:
                        save_rules_with_lock(rules_file, batch_rules)
                    except Exception as e:
                        print(f"{str(e)}")
            
            total_processed_count += batch_processed_count
            
            current_rules_count = get_current_rules_count("./rules.json")
            
            record_batch_processing(batch_num, total_processed_count, len(batch), current_rules_count, deleted_rules_count)

        return results
        
    except Exception as e:
        print(f"Error processing devices: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


async def main(input_file: str):
    await init_clients()
    
    output_file = './res_result.json'
    
    # Ensure output file is empty
    if Path(output_file).exists():
        Path(output_file).unlink()
    
    # Clear final_label.txt file to ensure fresh start each run
    final_label_path = "./final_label.txt"
    if Path(final_label_path).exists():
        Path(final_label_path).unlink()

    # Delete rule growth tracker file to ensure fresh start each run
    tracker_path = "./rules_growth_tracker.json"
    if Path(tracker_path).exists():
        Path(tracker_path).unlink()

    # Delete rules file to ensure fresh start each run
    rules_path = "./rules.json"
    if Path(rules_path).exists():
        Path(rules_path).unlink()
   
    results = await process_devices(input_file, output_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Process devices and extract rules")
    parser.add_argument("input_file", help="Path to input data JSON/JSONL file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import time
    time1 = time.time()
    asyncio.run(main(args.input_file))
    time2 = time.time()
    print(f"Total execution time: {time2-time1} seconds")
