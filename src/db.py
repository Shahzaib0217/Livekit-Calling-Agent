import os
import logging
from typing import Any, Dict, Optional
from src.db import create_client

from config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE,
)

logger = logging.getLogger(__name__)


async def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE:
        logger.error("Supabase credentials not available!")
        return None
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)


async def get_customer_by_phone(phone: str) -> Optional[Dict[str, Any]]:
    """
    Get customer information by phone number from Supabase.
    Returns customer data including first_name and last_name if found.
    """
    if not phone or not phone.strip():
        return None
    
    try:
        supabase = await get_supabase_client()
        response = supabase.table("customers").select("*").eq("phone", phone.strip()).execute()
        
        if response.data and len(response.data) > 0:
            customer = response.data[0]
            logger.info(f"Found customer: {customer.get('first_name', '')} {customer.get('last_name', '')} for phone {phone}")
            return customer
        else:
            logger.debug(f"No customer found for phone: {phone}")
            return None
            
    except Exception as e:
        logger.error(f"Error looking up customer by phone {phone}: {e}")
        return None