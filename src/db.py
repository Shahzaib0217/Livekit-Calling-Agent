import os
import logging
from typing import Any, Dict, List, Optional, Union, Callable, List
from supabase import create_client
from datetime import datetime

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

async def get_menu_by_phone(
    phone_number: str,
    process_product: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Load products for a business from Supabase using phone number.
    Returns a list of product dicts (empty list if none or on error).
    If `process_product` is provided, it will be called for each product (useful to pass `self._add_product_to_cache`).
    """
    products: List[Dict[str, Any]] = []

    if not phone_number or not phone_number.strip():
        logger.warning("get_menu_by_phone called with empty phone_number")
        return products

    try:
        supabase = await get_supabase_client()
        if not supabase:
            logger.error("No supabase client available")
            return products

        # 1) Get business by phone number
        biz_resp = supabase.table("businesses").select("id").eq("phone", phone_number.strip()).maybe_single().execute()

        if getattr(biz_resp, "error", None):
            logger.error("Supabase error while fetching business for %s: %s", phone_number, biz_resp.error)
            return products

        business = getattr(biz_resp, "data", None)
        if not business:
            logger.warning("No business found for phone number %s", phone_number)
            return products

        business_id = business.get("id")
        if not business_id:
            logger.warning("Business record missing 'id' for phone %s", phone_number)
            return products

        # 2) Get products for this business
        prod_resp = (
            supabase.table("products")
            .select("*")
            .eq("business_id", business_id)
            .eq("is_available", True)
            .execute()
        )

        if getattr(prod_resp, "error", None):
            logger.error("Supabase error while fetching products for business %s: %s", business_id, prod_resp.error)
            return products

        products = prod_resp.data or []

        logger.info("Found %d products for business_id=%s (phone=%s)", len(products), business_id, phone_number)

        # Optionally process each product (e.g., add to a cache)
        if process_product:
            for p in products:
                try:
                    process_product(p)
                except Exception as e:
                    logger.exception("Error processing product %s: %s", p.get("id", "<no-id>"), e)

        return products

    except Exception as e:
        logger.exception("Error loading products from Supabase for phone %s: %s", phone_number, e)
        return products



# -------------------------
# Menu helper utilities
# -------------------------
def build_compact_menu(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a compact menu structure from raw product records.
    Returns a dict: { "items": [...], "count": int, "fetched_at": iso }
    """
    items: List[Dict[str, Any]] = []
    for p in products or []:
        menu_json = p.get("menu_json") or {}
        price = p.get("base_price") or menu_json.get("base_price_kd") or None
        short_code = p.get("short_code") or menu_json.get("item_id")
        items.append({
            "id": p.get("id"),
            "name": p.get("name") or menu_json.get("name_en"),
            "name_ar": p.get("name_ar") or menu_json.get("name_ar"),
            "description": p.get("description") or menu_json.get("description_en"),
            "description_ar": p.get("description_ar") or menu_json.get("description_ar"),
            "price": price,
            "short_code": short_code,
            "in_stock": p.get("is_available") if p.get("is_available") is not None else menu_json.get("in_stock", True),
            "image_url": p.get("image_url"),
            "category": p.get("category") or menu_json.get("category"),
            "category_ar": p.get("category_ar") or menu_json.get("category_ar"),
            "updated_at": p.get("updated_at"),
            # keep menu_json if you want deeper choice_sets later
            # "menu_json": menu_json,
        })
    return {"items": items, "count": len(items), "fetched_at": datetime.utcnow().isoformat()}


def summarize_menu(compact_menu: Dict[str, Any], top_n: int = 3, currency: str = "") -> str:
    """
    Short human-friendly summary suitable for a system prompt that includes
    both English and Arabic names/categories and prices.
    """
    if not compact_menu:
        return "No menu available."

    items = compact_menu.get("items", []) or []
    top = items[:max(0, int(top_n))]

    def fmt_price(p):
        if p is None:
            return ""
        try:
            pval = float(p)
            return f"{pval:.2f} {currency}".strip()
        except Exception:
            return str(p)

    lines = []
    for i in top:
        code = i.get("short_code") or (i.get("id") and i.get("id")[:8]) or "unknown"
        name_en = (i.get("name") or "").strip()
        name_ar = (i.get("name_ar") or "").strip()
        price_str = fmt_price(i.get("price"))
        price_suffix = f" — {price_str}" if price_str else ""

        if name_en and name_ar:
            lines.append(f"{code}: {name_en} / {name_ar}{price_suffix}")
        elif name_en:
            lines.append(f"{code}: {name_en}{price_suffix}")
        elif name_ar:
            lines.append(f"{code}: {name_ar}{price_suffix}")
        else:
            lines.append(f"{code}: Unnamed{price_suffix}")

    if not lines:
        items_summary = "No items available."
    else:
        items_summary = "; ".join(lines)
        if not items_summary.endswith("."):
            items_summary = items_summary + "."

    # categories in EN and AR
    categories_en = [c for c in {i.get("category") for i in items if i.get("category")} if c]
    categories_ar = [c for c in {i.get("category_ar") for i in items if i.get("category_ar")} if c]

    cat_en_str = ", ".join(categories_en[:10]) if categories_en else "None"
    cat_ar_str = ", ".join(categories_ar[:10]) if categories_ar else "None"

    top_count = len(top)
    return (
        f"Top {top_count} items: {items_summary} "
        f"Categories (EN): {cat_en_str}. Categories (AR): {cat_ar_str}."
    )



def find_item_by_code(compact_menu: Dict[str, Any], code: str) -> Optional[Dict[str, Any]]:
    """
    Find an item by short_code or id (case-insensitive for short_code).
    """
    if not compact_menu or not code:
        return None
    code_norm = code.strip().lower()
    for it in compact_menu.get("items", []):
        if not it:
            continue
        if (it.get("short_code") and it["short_code"].strip().lower() == code_norm) or (it.get("id") == code):
            return it
    return None