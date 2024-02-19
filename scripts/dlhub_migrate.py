from garden_ai import GardenClient, Garden
from garden_ai.entrypoints import (
    EntrypointMetadata,
    Repository,
    Paper,
    RegisteredEntrypoint,
    Step,
)
from garden_ai import local_data
from garden_ai.app.garden import add_entrypoint
from globus_compute_sdk.sdk.web_client import FunctionRegistrationData

from typing import List, Optional

from dataclasses import dataclass


@dataclass
class EntrypointBundle:
    entrypoint_meta: EntrypointMetadata
    dlhub_fid: str
    docker_hub_container_uri: str
    test_function: Optional[str]


the_only_step = Step(
    function_name="dlhub_run",
    description="No function text available - this model was migrated from DLHub.",
    function_text="N/A",
)

# First, make EntrypointMetadata and Garden objects for all of the entrypoints and gardens.
# ^ done
# Second, make a function that takes an EntrypointMetadata, a function ID, and a container location.
#   It should register a new globus compute function that points at the container.
#   It should then create and save the entrypoint locally.
# Third, create all the entrypoints.
# Fourth, create and save all the gardens locally. Add the entrypoints.
# Fifth, publish the gardens I guess!


def create_entrypoint(entrypoint_bundle: EntrypointBundle):
    gc = GardenClient()
    function_code = gc.compute_client.get_function(entrypoint_bundle.dlhub_fid)[
        "function_code"
    ]
    registered_container_id = gc.compute_client.register_container(
        entrypoint_bundle.docker_hub_container_uri, "docker"
    )
    entrypoint_meta = entrypoint_bundle.entrypoint_meta
    # print(entrypoint_meta)
    frd = FunctionRegistrationData(
        function_code=function_code,
        function_name="dlhub_run",
        container_uuid=registered_container_id,
        description=f"Migrated version of DLHub model {entrypoint_meta.doi} for Garden",
        public=True,
        group=None,
    )
    resp = gc.compute_client.web_client.register_function(frd)
    new_fid = resp["function_uuid"]
    test_functions = (
        [entrypoint_bundle.test_function] if entrypoint_bundle.test_function else []
    )
    migrated_entrypoint = RegisteredEntrypoint(
        **entrypoint_meta.dict(),
        test_functions=test_functions,
        func_uuid=new_fid,
        container_uuid=registered_container_id,
        steps=[the_only_step],
    )
    local_data.put_local_entrypoint(migrated_entrypoint)
    return migrated_entrypoint


def create_entrypoints(entrypoint_bundles: List[EntrypointBundle]):
    for bundle in entrypoint_bundles:
        create_entrypoint(bundle)
        print(f"Created entrypoint with doi {bundle.entrypoint_meta.doi}")


def create_gardens(gardens: List[Garden]):
    for garden in gardens:
        local_data.put_local_garden(garden)
        print(f"Saved garden with doi {garden.doi}")


#
# STANDALONE MODELS
#

pacbed_entrypoint_meta = EntrypointMetadata(
    doi="10.26311/cd31-az33",
    title="PACBED-CNN: Infer thickness and mistilt from position-averaged convergent beam electron diffraction patterns",
    authors=[
        "Michael Oberaigner",
        "Alexander Clausen",
        "Dieter Weber",
        "Gerald Kothleitner",
        "Rafal E Dunin-Borkowski",
        "Daniel Knez",
    ],
    short_name="pacbed_infer",
    description='Input: (Type: dict) PACBED data array at key "pacbed", and parameters including "acceleration_voltage_kV", "zone_u","zone_v","zone_w", "crystal_structure", "convergence_angle_mrad". Output: (Type: dict) Predicted thickness in angstrom, thickness CNN output, mistilt angle in mrad, mistilt CNN output, and scaling factor used for preprocessing the PACBED.',
    year="2023",
    tags=["Materials Science"],
    models=[],
    repositories=[
        Repository(
            url="https://github.com/MichaelO1993/PACBED-CNN",
            repo_name="PACBED-CNN",
            contributors=[],
        )
    ],
    papers=[
        Paper(
            citation="Michael Oberaigner, Alexander Clausen, Dieter Weber, Gerald Kothleitner, Rafal E Dunin-Borkowski, Daniel Knez, Online Thickness Determination with Position Averaged Convergent Beam Electron Diffraction using Convolutional Neural Networks, Microscopy and Microanalysis, Volume 29, Issue 1, February 2023, Pages 427–436, https://doi.org/10.1093/micmic/ozac050",
            authors=[
                "Michael Oberaigner",
                "Alexander Clausen",
                "Dieter Weber",
                "Gerald Kothleitner",
                "Rafal E Dunin-Borkowski",
                "Daniel Knez",
            ],
            title="Online Thickness Determination with Position Averaged Convergent Beam Electron Diffraction using Convolutional Neural Networks",
            doi="10.1093/micmic/ozac050",
        )
    ],
)

pacbed_bundle = EntrypointBundle(
    pacbed_entrypoint_meta,
    "35090f12-ec2c-4138-8433-2bbdb54768e1",
    "gardenai/cbc3e248-8b62-4c74-b0ba-ee837197c3be",
    None,
)

defect_track_entrypoint = EntrypointMetadata(
    doi="10.26311/aefd-p769",
    title="DefectTrack: multi-object tracking algorithm for quantitative defect analysis of in-situ TEM videos in real-time",
    authors=[
        "Rajat Sainju",
        "Wei-Ying Chen",
        "Samuel Schaefer",
        "Qian Yang",
        "Caiwen Ding",
        "Meimei Li",
        "Yuanyuan Zhu",
    ],
    short_name="detect_defect_clisters",
    description="Input: (Type: dict) 8-bit image array list and OPTIONAL hyperparameters. Output: (Type: pandas dataframe) panda dataframe containing info of detected defects in each frame.",
    year="2023",
    tags=["Materials Science"],
    models=[],
    repositories=[],
    papers=[
        Paper(
            citation="Sainju, R., Chen, WY., Schaefer, S. et al. DefectTrack: a deep learning-based multi-object tracking algorithm for quantitative defect analysis of in-situ TEM videos in real-time. Sci Rep 12, 15705 (2022). https://doi.org/10.1038/s41598-022-19697-1",
            authors=[
                "Rajat Sainju",
                "Wei-Ying Chen",
                "Samuel Schaefer",
                "Qian Yang",
                "Caiwen Ding",
                "Meimei Li",
                "Yuanyuan Zhu",
            ],
            title="DefectTrack: a deep learning-based multi-object tracking algorithm for quantitative defect analysis of in-situ TEM videos in real-time",
            doi="10.1038/s41598-022-19697-1",
        )
    ],
)

defect_track_bundle = EntrypointBundle(
    defect_track_entrypoint,
    "ff684740-98fb-4f6f-afb6-443388ed7d6e",
    "gardenai/0384c964-4ba3-44b5-9340-f4c63c3d1ec2",
    None,
)

semiconductor_impurity_entrypoint = EntrypointMetadata(
    doi="10.26311/3hz8-as26",
    title="Semiconductor defect impurity levels",
    authors=[
        "Maciej Polak",
        "Ryan Jacobs",
        "Arun Mannodi-Kanakkithodi",
        "Maria Chan",
        "Dane Morgan",
    ],
    short_name="predict_defect_level_energies",
    description="Input: (Type: ndarray Shape: ['None', '15']) List of 15 elemental and one-hot encoded features to evaluate model. The list includes: M_3site, M_i_3site, M_i_neut_site, M_i_5site, M_5site, charge_from, charge_to, epsilon, CovalentRadius_max_value, ElectronAffinity_composition_average, NUnfilled_difference, phi_arithmetic_average, Site1_AtomicRadii_arithmetic_average, Site1_BCCvolume_padiff_differenc, Site1_HHIr_composition_average. Output: (Type: ndarray, Shape: 'None') Predictions of semiconductor defect level energies (in eV)",
    year="2020",
    tags=["Materials Science"],
    models=[],
    repositories=[],
    papers=[],
)

semiconductor_test_function = """def test_model():
    import numpy as np
    # The input shape is n rows of 15 attributes - replace with real data
    input = np.zeros((1, 15))
    return predict_defect_level_energies(input)
"""

semiconductor_impurity_bundle = EntrypointBundle(
    semiconductor_impurity_entrypoint,
    "6edf2522-75de-423c-adc9-fd24371550db",
    "gardenai/8db49340-1963-4ce5-a009-28cd4448bb45",
    semiconductor_test_function,
)

standalone_entrypoint_bundles = [
    pacbed_bundle,
    defect_track_bundle,
    semiconductor_impurity_bundle,
]

#
# ATOM SEGRESNET MODELS
#


def segmentation_test_function(fn_name: str):
    first_part = """def test_model():
    with requests.get('https://zenodo.org/record/10672182/files/testimage.npy') as r:
        r.raise_for_status()  # Ensure the download was successful
        with open('testimpage.npy', 'wb') as f:
            f.write(r.content)
    img = np.load('testimage.npy')
    input = {
        'image': img,
        'modelweights': 'gaussianMask+',
        'cuda': False,
        'change_size': 2
    }
"""
    return first_part + f"    return {fn_name}(input)\n"


lin_entrypoint_meta = EntrypointMetadata(
    doi="10.26311/8s9h-dz64",
    title="Lin AtomSegNet",
    authors=[
        "Ruoqian Lin",
        "Rui Zhang",
        "Chunyang Wang",
        "Xiao-Qing Yang",
        "Huolin Xin",
        "Jingrui Wei",
        "M Paul Voyles",
    ],
    short_name="locate_atomic_columns_lin",
    description="Input: (Type: dict) Dict of input image array and resize factor, or a list of dicts. Output: (Type: ndarray) List of the coordinates of located atomic columns for all the inputs",
    year="2022",
    tags=["Materials Science"],
    models=[],
    repositories=[],
    papers=[
        Paper(
            citation="Sainju, R., Chen, WY., Schaefer, S. et al. DefectTrack: a deep learning-based multi-object tracking algorithm for quantitative defect analysis of in-situ TEM videos in real-time. Sci Rep 12, 15705 (2022). https://doi.org/10.1038/s41598-022-19697-1",
            authors=[
                "Jingrui Wei",
                "Ben Blaiszik",
                "Aristana Scourtas",
                "Dane Morgan",
                "Paul M Voyles",
            ],
            title="Benchmark tests of atom segmentation deep learning models with a consistent dataset",
            doi="10.48550/arXiv.2207.10173",
        )
    ],
)
lin_bundle = EntrypointBundle(
    lin_entrypoint_meta,
    "8d8e379f-032b-4435-971e-2a00a1445a10",
    "gardenai/f80bdddc-4aee-480a-983a-bc5c6c28fb71",
    segmentation_test_function(lin_entrypoint_meta.short_name),
)
maxim_entrypoint_meta = EntrypointMetadata(
    doi="10.26311/bf7a-7071",
    title="Ziatdinov AtomNet",
    authors=["Ziatdinov Maxim", "Jingrui Wei"],
    short_name="locate_atomic_columns_ziatdinov",
    description="Input: (Type: dict) Dict of input image array and resize factor, or a list of dicts. Output: (Type: ndarray) List of the coordinates of located atomic columns for all the inputs",
    year="2022",
    tags=["Materials Science"],
    models=[],
    repositories=[],
    papers=[
        Paper(
            citation="Sainju, R., Chen, WY., Schaefer, S. et al. DefectTrack: a deep learning-based multi-object tracking algorithm for quantitative defect analysis of in-situ TEM videos in real-time. Sci Rep 12, 15705 (2022). https://doi.org/10.1038/s41598-022-19697-1",
            authors=[
                "Jingrui Wei",
                "Ben Blaiszik",
                "Aristana Scourtas",
                "Dane Morgan",
                "Paul M Voyles",
            ],
            title="Benchmark tests of atom segmentation deep learning models with a consistent dataset",
            doi="10.48550/arXiv.2207.10173",
        )
    ],
)
maxim_bundle = EntrypointBundle(
    maxim_entrypoint_meta,
    "929b6121-2bc6-40b4-abca-7e8a96d9bd62",
    "gardenai/eeb4ee9c-1696-4a57-9836-83aaa91b7334",
    None,
)

crystal_lattice_entrypoint_meta = EntrypointMetadata(
    doi="10.26311/e2mw-qf63",
    title="Atomai SegResNet trained on 5 crystal lattices",
    authors=["Ziatdinov Maxim", "Jingrui Wei"],
    short_name="locate_atomic_columns_crystal_lattice",
    description="Input: (Type: dict) Dict of input image array and resize factor, or a list of dicts. Output: (Type: ndarray) List of the coordinates of located atomic columns for all the inputs",
    year="2022",
    tags=["Materials Science"],
    models=[],
    repositories=[],
    papers=[
        Paper(
            citation="Sainju, R., Chen, WY., Schaefer, S. et al. DefectTrack: a deep learning-based multi-object tracking algorithm for quantitative defect analysis of in-situ TEM videos in real-time. Sci Rep 12, 15705 (2022). https://doi.org/10.1038/s41598-022-19697-1",
            authors=[
                "Jingrui Wei",
                "Ben Blaiszik",
                "Aristana Scourtas",
                "Dane Morgan",
                "Paul M Voyles",
            ],
            title="Benchmark tests of atom segmentation deep learning models with a consistent dataset",
            doi="10.48550/arXiv.2207.10173",
        )
    ],
)

crystal_lattice_bundle = EntrypointBundle(
    crystal_lattice_entrypoint_meta,
    "7c5f467c-76e9-49a5-bb11-9388296a7da0",
    "gardenai/f8159f91-1661-438e-a580-3b8e45d54886",
    segmentation_test_function(crystal_lattice_entrypoint_meta.short_name),
)

silicon_entrypoint_meta = EntrypointMetadata(
    doi="10.26311/b6zb-ns88",
    title="Atomai SegResNet trained on Si[110]",
    authors=["Ziatdinov Maxim", "Jingrui Wei"],
    short_name="locate_atomic_columns_silicon",
    description="Input: (Type: dict) Dict of input image array and resize factor, or a list of dicts. Output: (Type: ndarray) List of the coordinates of located atomic columns for all the inputs",
    year="2022",
    tags=["Materials Science"],
    models=[],
    repositories=[],
    papers=[
        Paper(
            citation="Sainju, R., Chen, WY., Schaefer, S. et al. DefectTrack: a deep learning-based multi-object tracking algorithm for quantitative defect analysis of in-situ TEM videos in real-time. Sci Rep 12, 15705 (2022). https://doi.org/10.1038/s41598-022-19697-1",
            authors=[
                "Jingrui Wei",
                "Ben Blaiszik",
                "Aristana Scourtas",
                "Dane Morgan",
                "Paul M Voyles",
            ],
            title="Benchmark tests of atom segmentation deep learning models with a consistent dataset",
            doi="10.48550/arXiv.2207.10173",
        )
    ],
)

silicon_bundle = EntrypointBundle(
    silicon_entrypoint_meta,
    "c700ada9-8797-4134-8b39-43199dc7ccd4",
    "gardenai/4b4197ee-ad44-49f4-9981-de81264e8f9d",
    segmentation_test_function(silicon_entrypoint_meta.short_name),
)

sto_entrypoint_meta = EntrypointMetadata(
    doi="10.26311/q6e2-2p11",
    title="Atomai SegResNet trained on STO[100]",
    authors=["Ziatdinov Maxim", "Jingrui Wei"],
    short_name="locate_atomic_columns_sto",
    description="Input: (Type: dict) Dict of input image array and resize factor, or a list of dicts. Output: (Type: ndarray) List of the coordinates of located atomic columns for all the inputs",
    year="2022",
    tags=["Materials Science"],
    models=[],
    repositories=[],
    papers=[
        Paper(
            citation="Sainju, R., Chen, WY., Schaefer, S. et al. DefectTrack: a deep learning-based multi-object tracking algorithm for quantitative defect analysis of in-situ TEM videos in real-time. Sci Rep 12, 15705 (2022). https://doi.org/10.1038/s41598-022-19697-1",
            authors=[
                "Jingrui Wei",
                "Ben Blaiszik",
                "Aristana Scourtas",
                "Dane Morgan",
                "Paul M Voyles",
            ],
            title="Benchmark tests of atom segmentation deep learning models with a consistent dataset",
            doi="10.48550/arXiv.2207.10173",
        )
    ],
)

sto_bundle = EntrypointBundle(
    sto_entrypoint_meta,
    "2bcac507-bc87-4fc9-b1d5-694137290d2e",
    "gardenai/ca0386bf-c26b-48e0-8ac3-ede8e3edb45b",
    segmentation_test_function(sto_entrypoint_meta.short_name),
)

one_entrypoint_meta = EntrypointMetadata(
    doi="10.26311/bkk2-gc19",
    title="Atomai SegResNet_1 for 5-fold CV",
    authors=["Ziatdinov Maxim", "Jingrui Wei"],
    short_name="locate_atomic_columns_1",
    description="Input: (Type: dict) Dict of input image array and resize factor, or a list of dicts. Output: (Type: ndarray) List of the coordinates of located atomic columns for all the inputs",
    year="2022",
    tags=["Materials Science"],
    models=[],
    repositories=[],
    papers=[
        Paper(
            citation="Sainju, R., Chen, WY., Schaefer, S. et al. DefectTrack: a deep learning-based multi-object tracking algorithm for quantitative defect analysis of in-situ TEM videos in real-time. Sci Rep 12, 15705 (2022). https://doi.org/10.1038/s41598-022-19697-1",
            authors=[
                "Jingrui Wei",
                "Ben Blaiszik",
                "Aristana Scourtas",
                "Dane Morgan",
                "Paul M Voyles",
            ],
            title="Benchmark tests of atom segmentation deep learning models with a consistent dataset",
            doi="10.48550/arXiv.2207.10173",
        )
    ],
)

one_entrypoint_bundle = EntrypointBundle(
    one_entrypoint_meta,
    "40c49ec0-2182-4ecc-a1fe-af2b306f9bac",
    "gardenai/b9673f64-19ca-4526-b656-5c1bfbce57c7",
    segmentation_test_function(one_entrypoint_meta.short_name),
)

two_entrypoint_meta = EntrypointMetadata(
    doi="10.26311/k2bk-hw50",
    title="Atomai SegResNet_2 for 5-fold CV",
    authors=["Ziatdinov Maxim", "Jingrui Wei"],
    short_name="locate_atomic_columns_2",
    description="Input: (Type: dict) Dict of input image array and resize factor, or a list of dicts. Output: (Type: ndarray) List of the coordinates of located atomic columns for all the inputs",
    year="2022",
    tags=["Materials Science"],
    models=[],
    repositories=[],
    papers=[
        Paper(
            citation="Sainju, R., Chen, WY., Schaefer, S. et al. DefectTrack: a deep learning-based multi-object tracking algorithm for quantitative defect analysis of in-situ TEM videos in real-time. Sci Rep 12, 15705 (2022). https://doi.org/10.1038/s41598-022-19697-1",
            authors=[
                "Jingrui Wei",
                "Ben Blaiszik",
                "Aristana Scourtas",
                "Dane Morgan",
                "Paul M Voyles",
            ],
            title="Benchmark tests of atom segmentation deep learning models with a consistent dataset",
            doi="10.48550/arXiv.2207.10173",
        )
    ],
)

two_entrypoint_bundle = EntrypointBundle(
    two_entrypoint_meta,
    "6e19bbce-9dc0-42a7-9fd8-7929f7a28569",
    "gardenai/bf6c6cbb-0d44-4f60-8569-01972fe98959",
    segmentation_test_function(two_entrypoint_meta.short_name),
)

three_entrypoint_meta = EntrypointMetadata(
    doi="10.26311/bgb7-k519",
    title="Atomai SegResNet_3 for 5-fold CV",
    authors=["Ziatdinov Maxim", "Jingrui Wei"],
    short_name="locate_atomic_columns_3",
    description="Input: (Type: dict) Dict of input image array and resize factor, or a list of dicts. Output: (Type: ndarray) List of the coordinates of located atomic columns for all the inputs",
    year="2022",
    tags=["Materials Science"],
    models=[],
    repositories=[],
    papers=[
        Paper(
            citation="Sainju, R., Chen, WY., Schaefer, S. et al. DefectTrack: a deep learning-based multi-object tracking algorithm for quantitative defect analysis of in-situ TEM videos in real-time. Sci Rep 12, 15705 (2022). https://doi.org/10.1038/s41598-022-19697-1",
            authors=[
                "Jingrui Wei",
                "Ben Blaiszik",
                "Aristana Scourtas",
                "Dane Morgan",
                "Paul M Voyles",
            ],
            title="Benchmark tests of atom segmentation deep learning models with a consistent dataset",
            doi="10.48550/arXiv.2207.10173",
        )
    ],
)

three_bundle = EntrypointBundle(
    three_entrypoint_meta,
    "6d2157b9-590d-4df7-a82a-e8db2fca3304",
    "gardenai/0240f2bb-5394-4452-b1b5-7efc79ebb87b",
    segmentation_test_function(three_entrypoint_meta.short_name),
)

four_entrypoint_meta = EntrypointMetadata(
    doi="10.26311/x13g-7f17",
    title="Atomai SegResNet_4 for 5-fold CV",
    authors=["Ziatdinov Maxim", "Jingrui Wei"],
    short_name="locate_atomic_columns_4",
    description="Input: (Type: dict) Dict of input image array and resize factor, or a list of dicts. Output: (Type: ndarray) List of the coordinates of located atomic columns for all the inputs",
    year="2022",
    tags=["Materials Science"],
    models=[],
    repositories=[],
    papers=[
        Paper(
            citation="Sainju, R., Chen, WY., Schaefer, S. et al. DefectTrack: a deep learning-based multi-object tracking algorithm for quantitative defect analysis of in-situ TEM videos in real-time. Sci Rep 12, 15705 (2022). https://doi.org/10.1038/s41598-022-19697-1",
            authors=[
                "Jingrui Wei",
                "Ben Blaiszik",
                "Aristana Scourtas",
                "Dane Morgan",
                "Paul M Voyles",
            ],
            title="Benchmark tests of atom segmentation deep learning models with a consistent dataset",
            doi="10.48550/arXiv.2207.10173",
        )
    ],
)

four_bundle = EntrypointBundle(
    four_entrypoint_meta,
    "dd7f77ac-d185-4a24-82d7-774e9c185f80",
    "gardenai/634ec49b-fac2-4625-bcdf-257e4ab287a6",
    segmentation_test_function(four_entrypoint_meta.short_name),
)

five_entrypoint_meta = EntrypointMetadata(
    doi="10.26311/s8hf-3v65",
    title="Atomai SegResNet_5 for 5-fold CV",
    authors=["Ziatdinov Maxim", "Jingrui Wei"],
    short_name="locate_atomic_columns_5",
    description="Input: (Type: dict) Dict of input image array and resize factor, or a list of dicts. Output: (Type: ndarray) List of the coordinates of located atomic columns for all the inputs",
    year="2022",
    tags=["Materials Science"],
    models=[],
    repositories=[],
    papers=[
        Paper(
            citation="Sainju, R., Chen, WY., Schaefer, S. et al. DefectTrack: a deep learning-based multi-object tracking algorithm for quantitative defect analysis of in-situ TEM videos in real-time. Sci Rep 12, 15705 (2022). https://doi.org/10.1038/s41598-022-19697-1",
            authors=[
                "Jingrui Wei",
                "Ben Blaiszik",
                "Aristana Scourtas",
                "Dane Morgan",
                "Paul M Voyles",
            ],
            title="Benchmark tests of atom segmentation deep learning models with a consistent dataset",
            doi="10.48550/arXiv.2207.10173",
        )
    ],
)

five_bundle = EntrypointBundle(
    five_entrypoint_meta,
    "2b501e11-095a-446c-969b-984b9f7fd4a3",
    "gardenai/c9c7a366-8153-409a-b89b-22cbc4ca6a92",
    segmentation_test_function(five_entrypoint_meta.short_name),
)

segmentation_entrypoint_bundles = [
    lin_bundle,
    maxim_bundle,
    crystal_lattice_bundle,
    silicon_bundle,
    sto_bundle,
    one_entrypoint_bundle,
    two_entrypoint_bundle,
    three_bundle,
    four_bundle,
    five_bundle,
]

#
# GARDENS
#

gc = GardenClient()

seg_res_net_garden = gc.create_garden(
    title="Atom segmentation deep learning models",
    authors=["Will Engler"],
    description="A collection of models that identify atomic column coordinates in scanning transmission electron microscopy (STEM) images.",
    year="2022",
)

pacbed_garden = gc.create_garden(
    title="Models for processing position-averaged convergent beam electron diffraction images",
    authors=["Will Engler"],
    description="This garden just hosts the PACBED-CNN model for now, and may host other position-averaged convergent beam electron diffraction models in the future.",
    year="2023",
)
defect_track_garden = gc.create_garden(
    title="Transmission electron microscopy (TEM) video analysis models",
    authors=["Will Engler"],
    description="This garden just hosts the DefectTrack model for now, and may host other TEM video processing models in the future.",
    year="2023",
)
semiconductor_properties_garden = gc.create_garden(
    title="Semiconductor property prediction models",
    authors=["Will Engler"],
    description="A collection of models for predicting properties of semiconductors",
    year="2020",
)

if __name__ == "__main__":
    create_entrypoints(standalone_entrypoint_bundles)
    create_entrypoints(segmentation_entrypoint_bundles)
    create_gardens(
        [
            seg_res_net_garden,
            pacbed_garden,
            defect_track_garden,
            semiconductor_properties_garden,
        ]
    )

    # Couldn't get this to work. Just manually add the entrypoints to the gardens.
    # reg_pacbed_entrypoint = local_data.get_local_entrypoint_by_doi(pacbed_entrypoint_meta.doi)
    # reg_defect_track_entrypoint = local_data.get_local_entrypoint_by_doi(defect_track_entrypoint.doi)
    # reg_semiconductor_entrypoint = local_data.get_local_entrypoint_by_doi(semiconductor_impurity_entrypoint.doi)
    # for (g, e) in zip(
    #     [pacbed_garden, defect_track_garden, semiconductor_properties_garden],
    #     [reg_pacbed_entrypoint, reg_defect_track_entrypoint, reg_semiconductor_entrypoint],
    # ):
    #     add_entrypoint(g.doi, e.doi, entrypoint_alias=None)
    #     local_data.put_local_garden(g)
    # # Add the entrypoints to the right gardens
    # for bundle in segmentation_entrypoint_bundles:
    #     entrypoint = local_data.get_local_entrypoint_by_doi(bundle.entrypoint_meta.doi)
    #     add_entrypoint(seg_res_net_garden.doi, entrypoint.doi, entrypoint_alias=None)
    # # Out of the loop bc we only need to write once at the end
    # local_data.put_local_garden(seg_res_net_garden)

    print("Done!")
    print("Next, add the entrypoints to the gardens.")
    print("Then manually run `garden-ai garden publish garden` for all 4 gardens.")
