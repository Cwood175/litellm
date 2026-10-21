from enum import Enum


class AzureCredentialType(str, Enum):
    CLIENT_SECRET = "ClientSecretCredential"
    MANAGED_IDENTITY = "ManagedIdentityCredential"
    CERTIFICATE = "CertificateCredential"
