# generated by datamodel-codegen:
#   filename:  datacite_4.3_schema.json
#   timestamp: 2023-01-26T18:29:42+00:00

from __future__ import annotations

from enum import Enum
from typing import Any, List, Union

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    confloat,
    field_validator,
)

from garden_ai.utils.pydantic import const_item_validator

from .schema_utils import UniqueList


class _DataCiteBaseModel(BaseModel, frozen=True):
    pass


class Identifier(_DataCiteBaseModel):
    identifier: str
    identifierType: str


class Subject(_DataCiteBaseModel):
    subject: str
    subjectScheme: str | None = None
    schemeUri: AnyUrl | None = None
    valueUri: AnyUrl | None = None
    lang: str | None = None


class AlternateIdentifier(_DataCiteBaseModel):
    alternateIdentifier: str
    alternateIdentifierType: str


class RightsListItem(_DataCiteBaseModel):
    rights: str | None = None
    rightsUri: AnyUrl | None = None
    rightsIdentifier: str | None = None
    rightsIdentifierScheme: str | None = None
    schemeUri: AnyUrl | None = None
    lang: str | None = None


class Container(_DataCiteBaseModel):
    type: str | None = None
    title: str | None = None
    firstPage: str | None = None


class NameType(str, Enum):
    Organizational = "Organizational"
    Personal = "Personal"


class NameIdentifier(_DataCiteBaseModel):
    nameIdentifier: str
    nameIdentifierScheme: str
    schemeUri: AnyUrl | None = None


class NameIdentifiers(RootModel):
    root: UniqueList[NameIdentifier]


class Affiliation(_DataCiteBaseModel):
    name: str
    affiliationIdentifier: str | None = None
    affiliationIdentifierScheme: str | None = None
    schemeUri: AnyUrl | None = None


class Affiliations(RootModel):
    root: UniqueList[Affiliation]


class TitleType(str, Enum):
    AlternativeTitle = "AlternativeTitle"
    Subtitle = "Subtitle"
    TranslatedTitle = "TranslatedTitle"
    Other = "Other"


class ContributorType(str, Enum):
    ContactPerson = "ContactPerson"
    DataCollector = "DataCollector"
    DataCurator = "DataCurator"
    DataManager = "DataManager"
    Distributor = "Distributor"
    Editor = "Editor"
    HostingInstitution = "HostingInstitution"
    Producer = "Producer"
    ProjectLeader = "ProjectLeader"
    ProjectManager = "ProjectManager"
    ProjectMember = "ProjectMember"
    RegistrationAgency = "RegistrationAgency"
    RegistrationAuthority = "RegistrationAuthority"
    RelatedPerson = "RelatedPerson"
    Researcher = "Researcher"
    ResearchGroup = "ResearchGroup"
    RightsHolder = "RightsHolder"
    Sponsor = "Sponsor"
    Supervisor = "Supervisor"
    WorkPackageLeader = "WorkPackageLeader"
    Other = "Other"


class Date(RootModel):
    root: Union[Any, Any, Any, Any, Any, Any, Any, Any]


class DateType(str, Enum):
    Accepted = "Accepted"
    Available = "Available"
    Copyrighted = "Copyrighted"
    Collected = "Collected"
    Created = "Created"
    Issued = "Issued"
    Submitted = "Submitted"
    Updated = "Updated"
    Valid = "Valid"
    Withdrawn = "Withdrawn"
    Other = "Other"


class ResourceTypeGeneral(str, Enum):
    Audiovisual = "Audiovisual"
    Collection = "Collection"
    DataPaper = "DataPaper"
    Dataset = "Dataset"
    Event = "Event"
    Image = "Image"
    InteractiveResource = "InteractiveResource"
    Model = "Model"
    PhysicalObject = "PhysicalObject"
    Service = "Service"
    Software = "Software"
    Sound = "Sound"
    Text = "Text"
    Workflow = "Workflow"
    Other = "Other"


class RelatedIdentifierType(str, Enum):
    ARK = "ARK"
    arXiv = "arXiv"
    bibcode = "bibcode"
    DOI = "DOI"
    EAN13 = "EAN13"
    EISSN = "EISSN"
    Handle = "Handle"
    IGSN = "IGSN"
    ISBN = "ISBN"
    ISSN = "ISSN"
    ISTC = "ISTC"
    LISSN = "LISSN"
    LSID = "LSID"
    PMID = "PMID"
    PURL = "PURL"
    UPC = "UPC"
    URL = "URL"
    URN = "URN"
    w3id = "w3id"


class RelationType(str, Enum):
    IsCitedBy = "IsCitedBy"
    Cites = "Cites"
    IsSupplementTo = "IsSupplementTo"
    IsSupplementedBy = "IsSupplementedBy"
    IsContinuedBy = "IsContinuedBy"
    Continues = "Continues"
    IsDescribedBy = "IsDescribedBy"
    Describes = "Describes"
    HasMetadata = "HasMetadata"
    IsMetadataFor = "IsMetadataFor"
    HasVersion = "HasVersion"
    IsVersionOf = "IsVersionOf"
    IsNewVersionOf = "IsNewVersionOf"
    IsPreviousVersionOf = "IsPreviousVersionOf"
    IsPartOf = "IsPartOf"
    HasPart = "HasPart"
    IsReferencedBy = "IsReferencedBy"
    References = "References"
    IsDocumentedBy = "IsDocumentedBy"
    Documents = "Documents"
    IsCompiledBy = "IsCompiledBy"
    Compiles = "Compiles"
    IsVariantFormOf = "IsVariantFormOf"
    IsOriginalFormOf = "IsOriginalFormOf"
    IsIdenticalTo = "IsIdenticalTo"
    IsReviewedBy = "IsReviewedBy"
    Reviews = "Reviews"
    IsDerivedFrom = "IsDerivedFrom"
    IsSourceOf = "IsSourceOf"
    IsRequiredBy = "IsRequiredBy"
    Requires = "Requires"
    IsObsoletedBy = "IsObsoletedBy"
    Obsoletes = "Obsoletes"


class DescriptionType(str, Enum):
    Abstract = "Abstract"
    Methods = "Methods"
    SeriesInformation = "SeriesInformation"
    TableOfContents = "TableOfContents"
    TechnicalInfo = "TechnicalInfo"
    Other = "Other"


class Longitude(RootModel):
    root: confloat(ge=-180.0, le=180.0)  # type: ignore


class Latitude(RootModel):
    root: confloat(ge=-90.0, le=90.0)  # type: ignore


class FunderIdentifierType(str, Enum):
    ISNI = "ISNI"
    GRID = "GRID"
    Crossref_Funder_ID = "Crossref Funder ID"
    ROR = "ROR"
    Other = "Other"


class Types(_DataCiteBaseModel):
    resourceType: str
    resourceTypeGeneral: ResourceTypeGeneral


class Creator(_DataCiteBaseModel):
    name: str
    nameType: NameType | None = None
    givenName: str | None = None
    familyName: str | None = None
    nameIdentifiers: NameIdentifiers | None = None
    affiliation: Affiliations | None = None
    lang: str | None = None


class Title(_DataCiteBaseModel):
    title: str
    titleType: TitleType | None = None
    lang: str | None = None


class Contributor(_DataCiteBaseModel):
    contributorType: ContributorType
    name: str
    nameType: NameType | None = None
    givenName: str | None = None
    familyName: str | None = None
    nameIdentifiers: NameIdentifiers | None = None
    affiliation: Affiliations | None = None
    lang: str | None = None


class DateModel(_DataCiteBaseModel):
    date: Date
    dateType: DateType
    dateInformation: str | None = None


class RelatedIdentifier(_DataCiteBaseModel):
    relatedIdentifier: str
    relatedIdentifierType: RelatedIdentifierType
    relationType: RelationType
    relatedMetadataScheme: str | None = None
    schemeUri: AnyUrl | None = None
    schemeType: str | None = None
    resourceTypeGeneral: ResourceTypeGeneral | None = None


class Description(_DataCiteBaseModel):
    description: str
    descriptionType: DescriptionType
    lang: str | None = None


class GeoLocationBox(_DataCiteBaseModel):
    westBoundLongitude: Longitude
    eastBoundLongitude: Longitude
    southBoundLatitude: Latitude
    northBoundLatitude: Latitude


class FundingReference(_DataCiteBaseModel):
    funderName: str
    funderIdentifier: str | None = None
    funderIdentifierType: FunderIdentifierType | None = None
    awardNumber: str | None = None
    awardUri: AnyUrl | None = None
    awardTitle: str | None = None


class GeoLocationPoint(_DataCiteBaseModel):
    pointLongitude: Longitude
    pointLatitude: Latitude


class GeoLocationPolygon(_DataCiteBaseModel):
    polygonPoints: List[GeoLocationPoint] = Field(..., min_length=4)
    inPolygonPoint: GeoLocationPoint | None = None


class GeoLocation(_DataCiteBaseModel):
    geoLocationPlace: str | None = None
    geoLocationPoint: GeoLocationPoint | None = None
    geoLocationBox: GeoLocationBox | None = None
    geoLocationPolygons: UniqueList[GeoLocationPolygon] | None = None


class DataciteSchema(_DataCiteBaseModel):
    model_config = ConfigDict(extra="forbid")

    # tweaked identifiers, no longer requires at least one. (only change from generated code)
    types: Types
    identifiers: UniqueList[Identifier]
    creators: UniqueList[Creator]
    titles: UniqueList[Title]
    publisher: str
    publicationYear: str
    subjects: UniqueList[Subject] | None = None
    contributors: UniqueList[Contributor] | None = None
    dates: UniqueList[DateModel] | None = None
    language: str | None = None
    alternateIdentifiers: UniqueList[AlternateIdentifier] | None = None
    relatedIdentifiers: UniqueList[RelatedIdentifier] | None = None
    sizes: UniqueList[str] | None = None
    formats: UniqueList[str] | None = None
    version: str | None = None
    rightsList: UniqueList[RightsListItem] | None = None
    descriptions: UniqueList[Description] | None = None
    geoLocations: UniqueList[GeoLocation] | None = None
    fundingReferences: UniqueList[FundingReference] | None = None
    schemaVersion: str = "http://datacite.org/schema/kernel-4"
    container: Container | None = None

    @field_validator("schemaVersion")
    @classmethod
    def const(cls, v, info):
        return const_item_validator(cls, v, info)
