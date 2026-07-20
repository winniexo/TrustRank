import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


import streamlit as st

from assets.components import (
    load_css,
    header,
    navbar,
    sidebar,
    product_card,
    footer
)

from src.search import search
print("search imported.")

st.set_page_config(
     page_title="TrustRank",
    page_icon="🛡",
    layout="wide")

load_css()

query,clicked = header()

navbar()
left, right = st.columns([1,4])

with left:
    rating = sidebar()

with right:

    if clicked and query:

        results = search(query)

        if results is None:

            st.error("No matching products found.")

        else:

            results = results[
                results["rating"] >= rating
            ]

            if results.empty:

                st.warning("No products match your filters.")

            else:

                for _, product in results.head(10).iterrows():

                    product_card(product)


footer()