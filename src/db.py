import os
import logging
from typing import Any, Dict, List, Optional, Union
from supabase import create_client

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


async def get_product_by_id(product_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a product from the database by ID.
    Ensures the product is available.
    """
    try:
        supabase = await get_supabase_client()
        response = supabase.table("products").select("*").eq("id", product_id).eq("is_available", True).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        else:
            logger.debug(f"Product not found or unavailable: {product_id}")
            return None
    except Exception as e:
        logger.error(f"Error fetching product {product_id}: {e}")
        return None


async def create_cart(location_id: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialize a new cart object.
    This is kept in memory until checkout.
    """
    cart = {
        "location_id": location_id,
        "customer_id": customer_id,
        "items": [],
        "subtotal": 0.0,
        "tax_amount": 0.0,
        "discount_amount": 0.0,
        "total_amount": 0.0,
    }
    logger.info(f"Created new cart for location={location_id}, customer={customer_id}")
    return cart


async def update_cart(
    cart: Dict[str, Any],
    product_id: Optional[str] = None,
    quantity: Optional[int] = None,
    selected_options: Optional[List[Dict[str, str]]] = None,
    special_instructions: Optional[str] = None,
    tax_rate: float = 0.0,
    discount_amount: float = 0.0
) -> Dict[str, Any]:
    """
    Update a cart by adding/updating items and recalculating totals.
    """
    if not cart:
        raise ValueError("Cart object is required")

    # If a product_id is provided, we add/update an item
    if product_id:
        product = await get_product_by_id(product_id)
        if not product:
            raise ValueError(f"Product {product_id} not found or unavailable")

        # Check if product already in cart → update quantity instead of duplicating
        existing_item = next((item for item in cart["items"] if item["product_id"] == product_id), None)
        unit_price = float(product["base_price"])

        if existing_item:
            if quantity:
                existing_item["quantity"] = quantity
            else:
                existing_item["quantity"] += 1

            existing_item["selected_options"] = selected_options or existing_item.get("selected_options", [])
            existing_item["special_instructions"] = special_instructions or existing_item.get("special_instructions", "")
            existing_item["total_price"] = existing_item["quantity"] * unit_price

        else:
            item = {
                "product_id": product_id,
                "name": product["name"],
                "quantity": quantity or 1,
                "unit_price": unit_price,
                "total_price": (quantity or 1) * unit_price,
                "selected_options": selected_options or [],
                "special_instructions": special_instructions or ""
            }
            cart["items"].append(item)

    # Always recalculate totals
    cart["subtotal"] = sum(item["total_price"] for item in cart["items"])
    cart["discount_amount"] = discount_amount
    cart["tax_amount"] = round(cart["subtotal"] * tax_rate, 2)
    cart["total_amount"] = round(cart["subtotal"] + cart["tax_amount"] - cart["discount_amount"], 2)

    logger.info(f"Updated cart: subtotal={cart['subtotal']}, total={cart['total_amount']}")
    return cart
