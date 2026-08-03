from pathlib import Path

import streamlit as st


# ------------------------------------------------------------------ #
# STYLES
# ------------------------------------------------------------------ #

def load_css():
    css_path = Path(__file__).parent / "style.css"
    css = css_path.read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# HEADER + NAV
# ------------------------------------------------------------------ #

def header():
    st.markdown(
        """
        <div class="tr-header">
            <div class="tr-logo">🛡️ Trust<span>Rank</span></div>
            <div class="tr-tagline">Find. Compare. Trust.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    search_col, button_col = st.columns([6, 1], vertical_alignment="center")

    with search_col:
        query = st.text_input(
            "Search",
            placeholder="Search for products \u2014 e.g. wireless earbuds, running shoes",
            label_visibility="collapsed",
            key="tr_search_query",
        )

    with button_col:
        clicked = st.button("Search", use_container_width=True, type="primary")

    return query, clicked


def navbar():
    """Renders the nav pill row. Returns the selected nav item."""
    choice = st.radio(
        "Navigate",
        ["Home", "Electronics", "Computers & Accessories", "Home & Kitchen", "About"],
        horizontal=True,
        label_visibility="collapsed",
        key="tr_nav",
    )
    st.markdown("<div class='tr-nav-divider'></div>", unsafe_allow_html=True)
    return choice


# ------------------------------------------------------------------ #
# SIDEBAR FILTERS
# ------------------------------------------------------------------ #

def sidebar():
    st.markdown("<div class='tr-filter-title'>Filters</div>", unsafe_allow_html=True)

    rating = st.slider("Minimum rating", 1.0, 5.0, 3.0, 0.5)


    price_range = st.slider(
        "Price range (\u20b9)", 0, 2500, (0, 2500), step=500
    )

    trusted_only = st.toggle("Highly trusted only", value=False)

    st.markdown("<div class='tr-filter-divider'></div>", unsafe_allow_html=True)

    return {
        "rating": rating,
        "price_range": price_range,
        "trusted_only": trusted_only,
    }


# ------------------------------------------------------------------ #
# TRUST RING (signature element)
# ------------------------------------------------------------------ #

def _trust_tier(score_out_of_10: float):
    if score_out_of_10 >= 7.5:
        return "#16A34A", "Highly Trusted"
    if score_out_of_10 >= 5.0:
        return "#FF8C00", "Moderate Trust"
    return "#DC2626", "High Risk"


def _trust_ring_html(score_out_of_10: float) -> str:
    score_out_of_10 = max(0.0, min(10.0, score_out_of_10))
    pct = score_out_of_10 * 10
    color, _ = _trust_tier(score_out_of_10)

    return f"""
    <div class="trust-ring" style="--pct:{pct}; --ring-color:{color};">
        <div class="trust-ring-inner">
            <div class="trust-ring-value">{score_out_of_10:.1f}</div>
            <div class="trust-ring-label">/ 10</div>
        </div>
    </div>
    """


# ------------------------------------------------------------------ #
# PRODUCT CARD
# ------------------------------------------------------------------ #

def product_card(product):
    """
    Renders a single product result.
    `product` is a pandas Series coming straight from the backend
    (src.search.search / re-ranking output). This function only reads
    fields defensively and never mutates or depends on backend internals
    beyond the columns already produced today.
    """

    

    if "trust_score" in product.index and product["trust_score"] is not None:
        trust_score = float(product["trust_score"]) * 10
    else:
        trust_score = float(product.get("rating", 3.0)) * 2

    trust_score = round(max(0.0, min(10.0, trust_score)), 1)
    ring_color, tier_label = _trust_tier(trust_score)

    fake_risk = max(5, 100 - int(trust_score * 10))
    verified = 100 - fake_risk

    if "confidence" in product.index and product["confidence"] is not None:
        confidence = round(float(product["confidence"]) * 100)
    else:
        confidence = round(trust_score * 10)

    with st.container(border=True):
        image_col, detail_col, trust_col = st.columns([1.1, 3.3, 1.4])

        # ---------------- IMAGE ---------------- #
        PLACEHOLDER_IMG = "https://placehold.co/300x300?text=No+Image"
        with image_col:
            img = str(product.get("img_link", "") or "").strip()
            if not img.startswith(("http://", "https://")):
                img = PLACEHOLDER_IMG
            try:
                st.image(img, use_container_width=True)
            except Exception:
                st.image(PLACEHOLDER_IMG, use_container_width=True)

        # ---------------- DETAILS ---------------- #
        with detail_col:
            st.markdown(f"##### {product.get('product_name', 'Untitled product')}")

            rating = float(product.get("rating", 0) or 0)
            stars = "\u2b50" * int(round(rating))
            st.markdown(f"<span class='tr-stars'>{stars}</span> **{rating:.1f}**", unsafe_allow_html=True)

            price_row = f"<span class='tr-price'>{product.get('discounted_price', '')}</span>"
            actual_price = product.get("actual_price", "")
            if actual_price:
                price_row += f" <span class='tr-price-strike'>{actual_price}</span>"
            st.markdown(price_row, unsafe_allow_html=True)

            about = str(product.get("about_product", "") or "")
            if about:
                st.write(about[:220] + ("..." if len(about) > 220 else ""))

            review = str(product.get("review_content", "") or "")
            if review:
                st.caption("\u201c" + review[:150] + ("..." if len(review) > 150 else "") + "\u201d")

            link = product.get("product_link", "")
            if link:
                st.link_button("Open on Amazon \u2197", link, use_container_width=False)

        # ---------------- TRUST CARD ---------------- #
        with trust_col:

            st.metric(
            "🛡 Trust Score",
            f"{trust_score:.1f}/10"
            )

            st.progress(trust_score / 10)

            st.success(tier_label)

            st.caption(f"Verified Reviews: {verified}%")

            st.caption(f"Fake Risk: {fake_risk}%")

            st.caption(f"Confidence: {confidence}%")

# ------------------------------------------------------------------ #
# FOOTER
# ------------------------------------------------------------------ #

def footer():
    st.markdown(
        """
        <div class="tr-footer">
            TrustRank \u2014 search engine with built-in fake review detection
        </div>
        """,
        unsafe_allow_html=True,
    )