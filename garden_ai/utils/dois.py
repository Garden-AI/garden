import requests


def is_doi_registered(doi):
    """
    Check if a DOI is registered by querying the DOI.org resolver.

    Parameters:
    doi (str): The DOI to check.

    Returns:
    bool: True if the DOI resolves successfully, False otherwise.
    """
    url = f"https://doi.org/{doi}"

    headers = {"Accept": "application/vnd.citationstyles.csl+json"}
    response = requests.get(url, headers=headers, allow_redirects=False)

    # Check if the response status code is a redirect (300-399), indicating the DOI is registered
    if 300 <= response.status_code < 400:
        return True
    else:
        return False
