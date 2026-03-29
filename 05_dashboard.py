# run this with:  streamlit run 05_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# page setup
st.set_page_config(
    page_title = "SMS Spam Explorer",
    page_icon  = "📱",
    layout     = "wide"
)

# loading data
@st.cache_data
def load():
    return pd.read_csv("outputs/spam_cleaned.csv")

data = load()
spam = data[data["label"] == "spam"]
ham  = data[data["label"] == "ham"]

# sidebar navigation
st.sidebar.title("📱 SMS Spam Explorer")
st.sidebar.markdown("Alok Chauhan & Aman Kumar")
st.sidebar.markdown("Batch 2C")
st.sidebar.markdown("---")

page = st.sidebar.radio("go to:", [
    "home",
    "charts",
    "word analysis",
    "segments",
    "check a message"
])

# ── HOME PAGE ────────────────────────────────────────────────
if page == "home":
    st.title("📱 SMS Spam Data Exploration")
    st.markdown("**Alok Chauhan & Aman Kumar | Batch 2C**")
    st.markdown("---")

    # big numbers at top
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Messages",     f"{len(data):,}")
    col2.metric("Spam Messages",      f"{len(spam):,}")
    col3.metric("Ham Messages",       f"{len(ham):,}")
    col4.metric("Duplicates Removed", "403")

    st.markdown("---")
    st.subheader("what we did")
    st.markdown("""
    - loaded the SMS spam dataset from UCI (5572 messages)
    - removed 403 duplicate messages
    - added 11 new feature columns
    - found patterns that separate spam from ham
    - built a rule based detection system
    """)

    st.subheader("top findings")
    st.success("phone numbers in messages = 99.7% spam rate")
    st.error("URLs are 57 times more common in spam")
    st.warning("3 or more spam signals = 100% spam rate")
    st.info("spam messages are 2 times longer than ham")

# ── CHARTS PAGE ──────────────────────────────────────────────
elif page == "charts":
    st.title("📊 EDA Charts")
    st.markdown("comparing spam and ham messages")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("message length")
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, color in [("spam","#E74C3C"),("ham","#2ECC71")]:
            d = data[data["label"] == label]["char_count"]
            ax.hist(d, bins=50, alpha=0.6,
                    color=color, label=label, density=True)
        ax.axvline(spam["char_count"].median(),
                   color="#E74C3C", linestyle="--", lw=2)
        ax.axvline(ham["char_count"].median(),
                   color="#2ECC71", linestyle="--", lw=2)
        ax.set_xlabel("characters")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("word count")
        fig, ax = plt.subplots(figsize=(7, 4))
        bp = ax.boxplot(
            [spam["word_count"], ham["word_count"]],
            tick_labels=["spam", "ham"],
            patch_artist=True
        )
        bp["boxes"][0].set_facecolor("#E74C3C")
        bp["boxes"][1].set_facecolor("#2ECC71")
        ax.set_ylabel("words")
        st.pyplot(fig)
        plt.close()

    st.subheader("feature comparison")
    names  = ["URL", "Number", "Prize", "FREE", "CALL", "TXT"]
    scols  = ["has_url","has_number","has_currency",
               "has_free","has_call","has_txt"]
    srates = [spam[c].mean()*100 for c in scols]
    hrates = [ham[c].mean()*100  for c in scols]

    table = pd.DataFrame({
        "feature"    : names,
        "spam %"     : [round(s, 1) for s in srates],
        "ham %"      : [round(h, 1) for h in hrates],
        "odds ratio" : [round(s/max(h,0.1), 1)
                        for s,h in zip(srates,hrates)]
    })
    st.dataframe(table, use_container_width=True)

# ── WORD ANALYSIS PAGE ───────────────────────────────────────
elif page == "word analysis":
    st.title("🔤 Word Frequency")

    STOPWORDS = [
        "i","me","my","we","our","you","your","he","him","his",
        "she","her","it","its","they","them","this","that","am",
        "is","are","was","were","be","been","have","has","had",
        "do","does","did","a","an","the","and","but","if","or",
        "of","at","by","for","with","to","from","in","on","not",
        "no","so","u","ur","r","ok","hi","hey","get","go","got"
    ]

    def get_words(message):
        message = str(message).lower()
        words   = message.split()
        clean   = []
        for word in words:
            word = word.strip(".,!?:;()[]\"'")
            if word not in STOPWORDS and len(word) > 2:
                clean.append(word)
        return clean

    n = st.slider("how many words to show?", 10, 30, 15)

    with st.spinner("counting words..."):
        spam_words = []
        for msg in spam["message"]:
            spam_words += get_words(msg)
        ham_words = []
        for msg in ham["message"]:
            ham_words += get_words(msg)
        spam_count = Counter(spam_words)
        ham_count  = Counter(ham_words)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("top spam words")
        top_spam = pd.Series(
            dict(spam_count.most_common(n))).sort_values()
        fig, ax = plt.subplots(figsize=(6, n * 0.4))
        ax.barh(top_spam.index, top_spam.values,
                color="#E74C3C")
        ax.set_xlabel("count")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("top ham words")
        top_ham = pd.Series(
            dict(ham_count.most_common(n))).sort_values()
        fig, ax = plt.subplots(figsize=(6, n * 0.4))
        ax.barh(top_ham.index, top_ham.values,
                color="#2ECC71")
        ax.set_xlabel("count")
        st.pyplot(fig)
        plt.close()

# ── SEGMENTS PAGE ────────────────────────────────────────────
elif page == "segments":
    st.title("🎯 Segmentation")
    st.markdown("spam rate in different message groups")

    avg = data["label_num"].mean() * 100

    choice = st.selectbox("pick a group:", [
        "message length",
        "spam signal score",
        "phone number",
        "exclamation marks"
    ])

    if choice == "message length":
        data2 = data.copy()
        data2["group"] = pd.cut(
            data2["char_count"],
            bins=[0, 50, 100, 160, 300, 9999],
            labels=["0-50","51-100","101-160","161-300","300+"]
        )
        seg   = data2.groupby("group", observed=True).agg(
            total=("label","count"),
            spam_count=("label_num","sum")
        )
        seg["spam_rate"] = (seg["spam_count"] /
                             seg["total"] * 100).round(1)
        title = "spam rate by message length"

    elif choice == "spam signal score":
        seg = data.groupby("spam_signals").agg(
            total=("label","count"),
            spam_count=("label_num","sum")
        )
        seg["spam_rate"] = (seg["spam_count"] /
                             seg["total"] * 100).round(1)
        seg.index = [f"{i} signals" for i in seg.index]
        title = "spam rate by signal score"

    elif choice == "phone number":
        seg = data.groupby("has_phone").agg(
            total=("label","count"),
            spam_count=("label_num","sum")
        )
        seg["spam_rate"] = (seg["spam_count"] /
                             seg["total"] * 100).round(1)
        seg.index = ["no phone", "has phone"]
        title = "spam rate by phone number"

    else:
        data2 = data.copy()
        data2["group"] = pd.cut(
            data2["exclamation"],
            bins=[-1, 0, 1, 2, 9999],
            labels=["0","1","2","3+"]
        )
        seg = data2.groupby("group", observed=True).agg(
            total=("label","count"),
            spam_count=("label_num","sum")
        )
        seg["spam_rate"] = (seg["spam_count"] /
                             seg["total"] * 100).round(1)
        title = "spam rate by exclamation marks"

    labels = [str(i) for i in seg.index]
    rates  = seg["spam_rate"].values
    totals = seg["total"].values
    colors = ["#E74C3C" if r > 50
              else "#e67e22" if r > 20
              else "#2ECC71" for r in rates]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, rates, color=colors, width=0.5)
    ax.axhline(avg, color="blue", linestyle="--",
               lw=2, label=f"overall avg: {avg:.1f}%")
    for bar, rate, total in zip(bars, rates, totals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{rate:.0f}%\n({total})",
                ha="center", fontweight="bold")
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("spam rate %")
    ax.legend()
    st.pyplot(fig)
    plt.close()
    st.dataframe(seg, use_container_width=True)

# ── CHECK A MESSAGE PAGE ─────────────────────────────────────
elif page == "check a message":
    st.title("🔍 Check Any Message")
    st.markdown("type a message and see if it looks like spam")

    msg = st.text_area("type your message here:",
                        height=120,
                        placeholder="e.g. FREE prize! Call 08001234567 NOW!")

    if msg:
        # checking each signal
        signals = {
            "has URL"           : "http" in msg.lower() or
                                   "www" in msg.lower(),
            "has phone number"  : any(len(w) >= 10 and w.isdigit()
                                       for w in msg.split()),
            "has prize words"   : any(w in msg.lower()
                                       for w in ["prize","cash","win"]),
            "has word FREE"     : "free" in msg.lower(),
            "has word CALL"     : "call" in msg.lower(),
            "has TXT or TEXT"   : "txt" in msg.lower() or
                                   "text" in msg.lower(),
            "has urgency words" : any(w in msg.lower()
                                       for w in ["urgent","claim","expire"]),
            "message is long"   : len(msg) > 100,
            "has exclamation"   : "!" in msg
        }

        score = sum(signals.values())

        # showing the score
        col1, col2, col3 = st.columns(3)
        col1.metric("characters", len(msg))
        col2.metric("words", len(msg.split()))
        col3.metric("spam score", f"{score}/9")

        # verdict
        if score >= 5:
            st.error(f"🚨 HIGH SPAM RISK - score {score}/9")
        elif score >= 3:
            st.warning(f"⚠️ MEDIUM SPAM RISK - score {score}/9")
        elif score >= 1:
            st.info(f"ℹ️ LOW SPAM RISK - score {score}/9")
        else:
            st.success("✅ LOOKS LIKE A NORMAL MESSAGE")

        # showing which signals triggered
        st.subheader("which signals triggered:")
        for signal, triggered in signals.items():
            if triggered:
                st.markdown(f"🔴 **{signal}** - yes found!")
            else:
                st.markdown(f"🟢 **{signal}** - not found")

    st.markdown("---")
    st.subheader("sample messages from dataset")
    n      = st.slider("how many to show?", 3, 10, 5)
    filter = st.radio("filter by:", ["all", "spam", "ham"])
    if filter == "all":
        sample = data.sample(n, random_state=42)
    else:
        sample = data[data["label"]==filter].sample(
            n, random_state=42)
    st.dataframe(
        sample[["label","message",
                "char_count","spam_signals"]].reset_index(drop=True),
        use_container_width=True
    )