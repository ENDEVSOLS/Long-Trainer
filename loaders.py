# from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import (CSVLoader, WikipediaLoader, UnstructuredURLLoader,
                                        YoutubeLoader, PyPDFLoader, BSHTMLLoader,
                                        Docx2txtLoader, UnstructuredMarkdownLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    def load_csv(self, path):
        """
        Load data from a CSV file at the specified path.

        Args:
            path (str): The file path to the CSV file.

        Returns:
            The loaded CSV data.

        Exceptions:
            Prints an error message if the CSV loading fails.
        """
        try:
            loader = CSVLoader(file_path=path)
            data = loader.load()
            return data
        except Exception as e:
            print(f"Error loading CSV: {e}")

    def wikipedia_query(self, search_query):
        """
        Query Wikipedia using a given search term and return the results.

        Args:
            search_query (str): The search term to query on Wikipedia.

        Returns:
            The query results.

        Exceptions:
            Prints an error message if the Wikipedia query fails.
        """
        try:
            data = WikipediaLoader(query=search_query, load_max_docs=2).load()
            return data
        except Exception as e:
            print(f"Error querying Wikipedia: {e}")

    def load_urls(self, urls):
        """
        Load and parse content from a list of URLs.

        Args:
            urls (list): A list of URLs to load.

        Returns:
            The loaded data from the URLs.

        Exceptions:
            Prints an error message if loading URLs fails.
        """
        try:
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
            return data
        except Exception as e:
            print(f"Error loading URLs: {e}")

    def load_YouTubeVideo(self, urls):
        """
        Load YouTube video information from provided URLs.

        Args:
            urls (list): A list of YouTube video URLs.

        Returns:
            The loaded documents from the YouTube URLs.

        Exceptions:
            Prints an error message if loading YouTube videos fails.
        """
        try:
            loader = YoutubeLoader.from_youtube_url(
                urls, add_video_info=True, language=["en", "pt", "zh-Hans", "es", "ur", "hi"],
                translation="en")
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error loading YouTube video: {e}")

    def load_pdf(self, path):
        """
        Load data from a PDF file at the specified path.

        Args:
            path (str): The file path to the PDF file.

        Returns:
            The loaded and split PDF pages.

        Exceptions:
            Prints an error message if the PDF loading fails.
        """
        try:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            return pages
        except Exception as e:
            print(f"Error loading PDF: {e}")

    def load_text_from_html(self, path):
        """
        Load and parse text content from an HTML file at the specified path.

        Args:
            path (str): The file path to the HTML file.

        Returns:
            The loaded HTML data.

        Exceptions:
            Prints an error message if loading text from HTML fails.
        """
        try:
            loader = BSHTMLLoader(path)
            data = loader.load()
            return data
        except Exception as e:
            print(f"Error loading text from HTML: {e}")

    def load_markdown(self, path):
        """
        Load data from a Markdown file at the specified path.

        Args:
            path (str): The file path to the Markdown file.

        Returns:
            The loaded Markdown data.

        Exceptions:
            Prints an error message if loading Markdown fails.
        """
        try:
            loader = UnstructuredMarkdownLoader(path)
            data = loader.load()
            return data
        except Exception as e:
            print(f"Error loading Markdown: {e}")

    def load_doc(self, path):
        """
        Load data from a DOCX file at the specified path.

        Args:
            path (str): The file path to the DOCX file.

        Returns:
            The loaded DOCX data.

        Exceptions:
            Prints an error message if loading DOCX fails.
        """
        try:
            loader = Docx2txtLoader(path)
            data = loader.load()
            return data
        except Exception as e:
            print(f"Error loading DOCX: {e}")

class TextSplitter:
    def __init__(self, chunk_size=1024, chunk_overlap=100):
        """
        Initialize the TextSplitter with a specific chunk size and overlap.

        Args:
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap size between chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_documents(self, documents):
        """
        Split the provided documents into chunks based on the chunk size and overlap.

        Args:
            documents (list): A list of documents to be split.

        Returns:
            A list of split documents.

        Exceptions:
            Prints an error message if splitting documents fails.
        """
        try:
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            print(f"Error splitting documents: {e}")
