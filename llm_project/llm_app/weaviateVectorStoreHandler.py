import weaviate
import weaviate.classes as wvc
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import pdfplumber
from langchain_core.documents import Document
from langchain_weaviate.vectorstores import WeaviateVectorStore
import re,os
from weaviate.classes.query import Filter
from .vectorMapper import add_mapping,get_mapping

weaviate_clien=None

def weaviate_store_init():
    global weaviate_clien
    weaviate_clien = weaviate.connect_to_local(
    )
    return weaviate_clien

def get_all_pdfs(directory):
    pdf_files = []
    # Traverse the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                # Join the directory path with the file name to get full path
                pdf_files.append(os.path.join(root, file))
    return pdf_files
    
def extract_pdf_content(pdf_paths):
    documents=[]
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extracting images
                thisDocument=Document(
                    page_content=page.extract_text(),
                    metadata={"filename":pdf_path}
                )
                documents.append(thisDocument)
    return documents
def is_table_of_contents(text):
    # Identify common ToC patterns: e.g., page numbers, dot leaders, or short sections
    toc_patterns = [
        r"\.{2,}",  # dot leader
        r"^\s*\d+\s*$",  # standalone page numbers
        r"^\s*第?\d+章",  # Chinese chapter heading (e.g., "第1章" or "1. Chapter")
        r"\.{2,}\s*\d+$",  # dot leader with page number at the end of the line
        r"^\s*目錄\s*$",  # detect "目錄" or similar ToC markers
    ]
    for pattern in toc_patterns:
        if re.search(pattern, text):
            return True
    return False

def filter_table_of_contents(documents):
    filtered_documents = []
    for doc in documents:
        content_lines = doc.page_content.splitlines()
        filtered_lines = [
            line for line in content_lines if not is_table_of_contents(line)
        ]
        filtered_content = "\n".join(filtered_lines)
        filtered_documents.append(
            Document(page_content=filtered_content, metadata=doc.metadata)
        )
    return filtered_documents
def filter_lines(document, keyword):
    pattern = re.compile(r".{2,}\s*\d+$")

    filtered_text = []
    for page in document:
        # Split the page text into lines
        lines = page.splitlines()
        # Filter out lines that contain the keyword
        filtered_lines = [line for line in lines if not pattern.match(line.strip())]
        # Join the filtered lines back into a string
        filtered_text.append("\n".join(filtered_lines))
    return filtered_text



def newVector(chat_id,pdf_file_Path):
    #to do
    global weaviate_clien
    if weaviate_clien is None or weaviate_clien.is_ready() is False:
        weaviate_store_init()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    filepath=get_all_pdfs(pdf_file_Path)
    documents=extract_pdf_content(filepath)
    print(f"檔案被切分為{len(documents)}塊")
    fd=filter_table_of_contents(documents)
    for thisfd in fd:
        if thisfd.page_content.strip() == "" or thisfd.page_content.strip() == "\n":
            try:
                fd.remove(thisfd)
            except:
                continue
    db = WeaviateVectorStore.from_documents(fd, embeddings, client=weaviate_clien)
    add_mapping(chat_id,db._index_name)

def queryVector(chat_id,message):
    dbindex=get_mapping(chat_id)
    global weaviate_clien
    if weaviate_clien is None or weaviate_clien.is_ready() is False:
        weaviate_store_init()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    db=WeaviateVectorStore(client=weaviate_clien,index_name=dbindex,text_key="text",embedding=embeddings)
    docs=db.similarity_search(message)
    theContext = ""
    for docPage in docs:
        thisFileName=docPage.metadata["filename"]
        theContext += docPage.page_content
        theContext += f"節錄自{thisFileName}\n"
    ragPromote = (
        f"\n以下是你參考的文件，將其統整併歸納後用於回答問題。這些參考資料使用者看不見，如果你要引用，需要將參考資料帶給使用者看。```\n{theContext} \n ```"
    )
    return ragPromote
    
def testloadfiles():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    filepath=get_all_pdfs("./testPath")
    documents=extract_pdf_content(filepath)
    print(f"檔案被切分為{len(documents)}塊")
    fd=filter_table_of_contents(documents)
    for thisfd in fd:
        if thisfd.page_content.strip() == "" or thisfd.page_content.strip() == "\n":
            try:
                fd.remove(thisfd)
            except:
                continue
    db = WeaviateVectorStore.from_documents(fd, embeddings, client=weaviate_clien)
    print(db._index_name)
    print(f"內容清理完成，剩下{len(fd)}塊")
    docs = db.similarity_search("itpet的主要目的是甚麼?")
    print(docs)



#weaviatetest.collections.delete_all()
