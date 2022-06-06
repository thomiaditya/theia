from fileinput import filename
from io import BytesIO, StringIO
from itertools import count
from google.cloud import storage
from google.oauth2 import service_account
import os
import glob

def get_byte_fileobj(project: str,
                     bucket: str,
                     path: str,
                     service_account_credentials_path: str = None) -> BytesIO:
    """
    Retrieve data from a given blob on Google Storage and pass it as a file object.
    :param path: path within the bucket
    :param project: name of the project
    :param bucket_name: name of the bucket
    :param service_account_credentials_path: path to credentials.
           TIP: can be stored as env variable, e.g. os.getenv('GOOGLE_APPLICATION_CREDENTIALS_DSPLATFORM')
    :return: file object (BytesIO)
    """
    blob = _get_blob(bucket, path, project, service_account_credentials_path)
    byte_stream = BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)
    return byte_stream

def get_bytestring(project: str,
                   bucket: str,
                   path: str,
                   service_account_credentials_path: str = None) -> bytes:
    """
    Retrieve data from a given blob on Google Storage and pass it as a byte-string.
    :param path: path within the bucket
    :param project: name of the project
    :param bucket_name: name of the bucket
    :param service_account_credentials_path: path to credentials.
           TIP: can be stored as env variable, e.g. os.getenv('GOOGLE_APPLICATION_CREDENTIALS_DSPLATFORM')
    :return: byte-string (needs to be decoded)
    """
    blob = _get_blob(bucket, path, project, service_account_credentials_path)
    s = blob.download_as_string()
    return s


def _get_blob(bucket_name, path, project, service_account_credentials_path):
    credentials = service_account.Credentials.from_service_account_file(
        service_account_credentials_path) if service_account_credentials_path else None
    storage_client = storage.Client(project=project, credentials=credentials)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(path)
    return blob

def get_directory_to_fileobj(project: str,
                                bucket: str,
                                path: str,
                                destination_path: str,
                                service_account_credentials_path: str = None) -> BytesIO:
    """
    Retrieve data from a given blob on Google Storage and pass it as a file object.
    :param path: path within the bucket
    :param project: name of the project
    :param bucket_name: name of the bucket
    :param service_account_credentials_path: path to credentials.
           TIP: can be stored as env variable, e.g. os.getenv('GOOGLE_APPLICATION_CREDENTIALS_DSPLATFORM')
    :return: file object (BytesIO)
    """

    storage_client = storage.Client(project=project, credentials=service_account.Credentials.from_service_account_file(service_account_credentials_path)) if service_account_credentials_path else None

    # Get the bucket
    bucket = storage_client.get_bucket(bucket)

    # Get the blobs
    blobs = bucket.list_blobs(prefix=path)

    # Loop through the blobs and download them
    count = 0
    for blob in blobs:
        if blob.name.endswith('/'):
            continue
        blob_list = blob.name.split('/')
        
        directory = os.sep.join(blob_list[:-1])
        filename = blob_list[-1]

        if not os.path.exists(os.path.join(destination_path, directory)):
            os.makedirs(os.path.join(destination_path, directory))

        full_path = os.path.join(destination_path, directory, filename)
        blob.download_to_filename(full_path)
        count += 1
    
    print("Downloaded {} files to {}".format(count, destination_path))

def upload_fileobj(project: str,
                   bucket: str,
                   directory_to_upload: str,
                   destination_path: str,
                   service_account_credentials_path: str = None) -> None:
    """
    Upload a file object to a given blob on Google Storage.
    :param path: path within the bucket
    :param project: name of the project
    :param bucket_name: name of the bucket
    :param fileobj: file object
    :param service_account_credentials_path: path to credentials.
           TIP: can be stored as env variable, e.g. os.getenv('GOOGLE_APPLICATION_CREDENTIALS_DSPLATFORM')
    :return: None
    """
    from alive_progress import alive_bar

    storage_client = storage.Client(project=project, credentials=service_account.Credentials.from_service_account_file(service_account_credentials_path)) if service_account_credentials_path else None

    # Get the bucket
    bucket = storage_client.get_bucket(bucket)

    # Loop through the blobs and download them
    count = 0
    for root, dirs, files in os.walk(directory_to_upload):
        for file in files:
            full_path = os.path.join(root, file)
            blob = bucket.blob(os.path.join(destination_path, full_path.replace(directory_to_upload + os.sep, '')).replace(os.sep, '/'))
            blob.upload_from_filename(full_path)
            count += 1
    
    print("Uploaded {} files to {}".format(count, destination_path))

# import os
# import dotenv
# dotenv.load_dotenv()
# credential = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
# get_directory_to_fileobj('zeta-resource-351216', 'theia-recommender', '.history', os.path.expanduser("~"), credential)
# upload_fileobj('zeta-resource-351216', 'theia-recommender', os.path.expanduser("~/.history"), 'test2-history', credential)