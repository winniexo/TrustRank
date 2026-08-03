import re

import streamlit as st

from assets.components import (
    load_css,
    header,
    navbar,
    sidebar,
    product_card,
    footer,
)
from src.search import search

st.set_page_config(
    page_title="TrustRank",
    page_icon=":shield:",
    layout="wide",
)

load_css()

# ------------------------------------------------------------------ #
# HEADER + SEARCH
# ------------------------------------------------------------------ #

query, clicked = header()

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if (clicked or query) and query.strip():
    st.session_state.search_query = query

query = st.session_state.search_query
nav_choice = navbar()

# If a nav pill other than Home is selected, treat it as a quick category
# shortcut for the category filter below (does not touch the backend).
nav_category_map = {
    "Electronics": "Electronics",
    "Computers & Accessories": "Computers&Accessories",
    "Home & Kitchen": "Home&Kitchen",
}

left, right = st.columns([1, 3.4], gap="large")

with left:
    filters = sidebar()

# ------------------------------------------------------------------ #
# FILTER HELPERS (frontend-only, does not modify the backend)
# ------------------------------------------------------------------ #


def _parse_price(value) -> float:
    """Best-effort parse of a price field like '\u20b91,299' into a float."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    digits = re.sub(r"[^\d.]", "", str(value))
    try:
        return float(digits) if digits else 0.0
    except ValueError:
        return 0.0


def _apply_frontend_filters(results, filters, nav_choice):
    filtered = results[results["rating"] >= filters["rating"]]

    # Category: only apply if the backend actually returned a category-like
    # column, so we never assume something the search pipeline doesn't give us.
    # "Home" and "About" are navigational, not product categories, so they
    # never filter results.
    category_cols = [c for c in ("category", "main_category") if c in filtered.columns]
    if nav_choice not in ("Home", "About") and category_cols:
        effective_category = nav_category_map.get(nav_choice, nav_choice)
        col = category_cols[0]
        filtered = filtered[
            filtered[col].astype(str).str.contains(effective_category, case=False, na=False)
        ]

    # Price range: only apply if a price column is present.
    price_col = "discounted_price" if "discounted_price" in filtered.columns else None
    if price_col:
        low, high = filters["price_range"]
        prices = filtered[price_col].apply(_parse_price)
        filtered = filtered[(prices >= low) & (prices <= high)]

    # Highly trusted toggle: only apply if a trust_score column exists.
    if filters["trusted_only"] and "trust_score" in filtered.columns:
        filtered = filtered[filtered["trust_score"].astype(float) * 10 >= 7.5]

    return filtered


# ------------------------------------------------------------------ #
# RESULTS
# ------------------------------------------------------------------ #

with right:
    if query:
        results = search(query)

        if results is None or results.empty:
            st.error("No matching products found. Try a different search term.")
        else:
            results = _apply_frontend_filters(results, filters, nav_choice)

            st.markdown("<div class='tr-results-heading'>Search Results</div>", unsafe_allow_html=True)

            if results.empty:
                st.markdown(
                    "<div class='tr-results-count'>0 results match your filters.</div>",
                    unsafe_allow_html=True,
                )
                st.warning("No products match your current filters. Try loosening them.")
            else:
                st.markdown(
                    f"<div class='tr-results-count'>{len(results)} result(s) for \u201c{query}\u201d</div>",
                    unsafe_allow_html=True,
                )

                LOW_RELEVANCE_THRESHOLD = 0.12
                if (
                    "similarity_score" in results.columns
                    and results["similarity_score"].max() < LOW_RELEVANCE_THRESHOLD
                ):
                    st.warning(
                        "These results have low text relevance to your search — "
                        "this catalog may not carry a strong match for "
                        f"\u201c{query}\u201d. Showing the closest matches anyway."
                    )

                for _, product in results.head(10).iterrows():
                    product_card(product)
    else:
        st.markdown("<div class='tr-results-heading'>Start a search</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='tr-results-count'>Search Amazon products above to see trust-ranked results.</div>",
            unsafe_allow_html=True,
        )

footer()