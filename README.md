# ENPH 353 Team 16 Main Repository
### Reese Critchlow and Tristan Lee
## Key Directory Tree
.
└── src
    ├── 2022_competition
    └── [controller_package](src/controller_package/)
        ├── launch
        ├── [models](src/controller_package/models/)
        │   ├── alpha_model.h5
        │   ├── license_model.h5
        │   ├── license_model_v2.h5
        │   ├── number_model.h5
        │   ├── parking_model.h5
        │   └── rm5_modified_10.h5 (driving model)
        ├── [nodes](src/controller_package/nodes/)
        │   ├── deprecated
        │   │   ├── data_collection.py
        │   │   ├── data_controller.py
        │   │   ├── imitation_controller.py
        │   │   ├── license_controller.py
        │   │   ├── license_identification.py
        │   │   └── pid_controller.py
        │   ├── main_controller.py
        │   ├── state_machine.py
        │   └── [utils](src/controller_package/nodes/utils)
        │       ├── direct_controller.py
        │       ├── image_processing.py
        │       └── timer_controller.py
        └── [scripts](src/controller_package/scripts)
            ├── frame_collector.py
            ├── turn_injector.py
            └── util
                ├── colour_finder.py
                └── image_renamer.py