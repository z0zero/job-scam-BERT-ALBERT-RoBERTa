from datetime import datetime
from typing import Any

import streamlit as st


def make_snippet(text: str, limit: int = 200) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "…"


def format_confidence(confidence: float) -> str:
    return f"{confidence * 100:.1f}%"


class HistoryView:
    @staticmethod
    def render(
        items: list[dict[str, Any]], offset: int, has_more: bool
    ) -> int | None:
        st.subheader("Analysis history")
        if not items:
            st.info("No analysis history yet.")
        for item in items:
            created = datetime.fromisoformat(
                item["created_at"].replace("Z", "+00:00")
            )
            heading = (
                f"{created:%Y-%m-%d %H:%M} — {item['prediction_label']} "
                f"({format_confidence(float(item['confidence']))})"
            )
            with st.expander(heading):
                st.caption(
                    f"Source: {item['input_source']} · "
                    f"Preview: {make_snippet(item['input_text'])}"
                )
                st.text_area(
                    "Analyzed text",
                    value=item["input_text"],
                    height=220,
                    disabled=True,
                    key=f"history_text_{item['id']}",
                )
                flags = item.get("red_flags") or []
                if flags:
                    st.markdown("**Red flags**")
                    for flag in flags:
                        st.write(f"- {flag}")
                else:
                    st.caption("No explicit heuristic red flags.")

        previous_col, next_col = st.columns(2)
        with previous_col:
            if st.button(
                "Previous",
                disabled=offset == 0,
                use_container_width=True,
            ):
                return max(0, offset - 20)
        with next_col:
            if st.button(
                "Next", disabled=not has_more, use_container_width=True
            ):
                return offset + 20
        return None

    @staticmethod
    def render_error(message: str) -> None:
        st.error(message)
