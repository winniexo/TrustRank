from pathlib import Path
import streamlit as st


def load_css():
    css_path = Path(__file__).parent / "assets" / "css"

    css = ""

    for file in css_path.glob("*.css"):
        css += file.read_text()

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def header():
    logo, search,button = st.columns([1,5,1])
    with logo:
        st.markdown("<h2>🛡 TrustRank</h2>", unsafe_allow_html=True)

    # Search Box
    with search:
        query = st.text_input(
            "",
            placeholder="Search for products...",
            label_visibility="collapsed"
        )

    # Search Button
    with button:
        clicked = st.button("Search")

    return query, clicked


def navbar():
    pass


def sidebar():
    st.header("Filters")

    rating = st.slider(
        "MINIMUM RATING",
        1.0,
        5.0,
        3.0
    )
    return rating


def product_card(product):
    left,right = st.columns([1,3])

    with left:
        st.image(
            product["img_link"],
            width=170
        )
    
    with right:
        st.subheader(product["product_name"])
    
        st.write(f"⭐ {product['rating']}")

        st.markdown(f"💰 {product['discounted_price']}")

        st.markdown( f"~~{product['actual_price']}~~")

        st.write( product["about_product"][:250] + "...")

        review = product["review_content"][:180]

        st.caption(review + " ")   

        st.link_button("🛒 Open on Amazon", product["product_link"])  

        st.divider()   
     


def footer():
    st.markdown("---")
    st.caption("TrustRank | powered by trusted rankings :)")