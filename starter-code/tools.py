import json
import os
from typing import List, Dict, Any
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw-data")

VALID_CATEGORIES = ("xe_dien", "du_lich")
VALID_PRIORITIES = ("low", "medium", "high")


def _load_json_list(path: str) -> List[Dict[str, Any]]:
    """Đọc file JSON dạng list; trả về [] nếu file rỗng hoặc hỏng."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    data = json.loads(content)
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.

    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.

    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    try:
        products = _load_json_list(catalog_file)
    except (json.JSONDecodeError, OSError) as e:
        return [{"error": f"Cannot read product catalog: {e}"}]

    if max_price is None:
        max_price = 999999999999
    category = (category or "").strip().lower()

    results = [
        p for p in products
        if str(p.get("category", "")).lower() == category
        and p.get("price_vnd", 0) <= max_price
    ]
    return sorted(results, key=lambda p: p.get("price_vnd", 0))


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.

    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.

    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    priority = (priority or "medium").strip().lower()
    if priority not in VALID_PRIORITIES:
        priority = "medium"

    # Load trước rồi mới append — tránh ghi đè ticket cũ (Trap 2)
    existing_tickets: List[Dict[str, Any]] = []
    if os.path.exists(tickets_file):
        try:
            existing_tickets = _load_json_list(tickets_file)
        except json.JSONDecodeError:
            return {"error": "Support tickets file is corrupted.", "status": "failed"}

    now = datetime.now()
    today = now.strftime("%Y%m%d")
    seq = len(existing_tickets) + 1
    ticket_id = f"TK-{today}-{seq:03d}"

    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": now.isoformat() + "+07:00",
        "category": "general"
    }
    existing_tickets.append(new_ticket)

    with open(tickets_file, "w", encoding="utf-8") as f:
        json.dump(existing_tickets, f, indent=2, ensure_ascii=False)

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công."
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": (
            "Tra cứu sản phẩm/dịch vụ Vingroup (xe điện VinFast, gói nghỉ dưỡng Vinpearl) "
            "theo danh mục và giá tối đa. Dùng khi khách muốn xem, tìm, so sánh hoặc hỏi giá sản phẩm."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm: 'xe_dien' (xe điện VinFast) hoặc 'du_lich' (resort/khách sạn Vinpearl).",
                    "enum": list(VALID_CATEGORIES)
                },
                "max_price": {
                    "type": "integer",
                    "description": "Giá tối đa tính bằng VNĐ (ví dụ 600 triệu = 600000000). Bỏ trống nếu không giới hạn.",
                    "minimum": 0
                }
            },
            "required": ["category"]
        }
    },
    {
        "name": "submit_support_ticket",
        "description": (
            "Tạo ticket hỗ trợ khi khách hàng báo lỗi, khiếu nại hoặc cần ghi nhận phản hồi. "
            "Chỉ gọi khi đã biết tên khách hàng và mô tả vấn đề."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ tên đầy đủ của khách hàng, ví dụ 'Lê Minh Khoa'."
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả ngắn gọn vấn đề khách hàng gặp phải."
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên: 'high' (nghiêm trọng/gấp), 'medium' (trung bình), 'low' (không gấp).",
                    "enum": list(VALID_PRIORITIES)
                }
            },
            "required": ["customer_name", "issue_description"]
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
