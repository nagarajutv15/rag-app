import re


def clean_documents(documents):

    cleaned_docs = []

    for doc in documents:

        text = doc.page_content

        # Remove multiple spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        # Remove repeated blank lines
        text = re.sub(
            r"\n{2,}",
            "\n",
            text
        )

        # Remove non-printable characters only
        text = re.sub(
            r"[\x00-\x1F\x7F]",
            "",
            text
        )

        text = text.strip()

        doc.page_content = text

        cleaned_docs.append(doc)

    return cleaned_docs