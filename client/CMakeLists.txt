cmake_minimum_required(VERSION 3.25)
project(client)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS
        Core
        Gui
        Widgets
        REQUIRED)

set(CMAKE_PREFIX_PATH /home/maneologu/libtorch/libtorch/share/cmake/Torch)
if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 52 61 75)
endif()
set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda)

find_package( Torch REQUIRED )

add_executable(client main.cpp UI/mainwindow.cpp UI/mainwindow.h UI/SettingsWindow.cpp UI/SettingsWindow.h models/CanvasModel.cpp models/CanvasModel.h views/CanvasView.cpp views/CanvasView.h models/GeneratedImagesModel.cpp models/GeneratedImagesModel.h views/GeneratedImagesView.cpp views/GeneratedImagesView.h controllers/MainController.cpp controllers/MainController.h settings/Settings.cpp settings/Settings.h strategies/generative_strategies/GenerativeStrategy.cpp strategies/generative_strategies/GenerativeStrategy.h strategies/generative_strategies/DefaultStrategy.cpp strategies/generative_strategies/DefaultStrategy.h strategies/generative_strategies/SketchyGanStrategy.cpp strategies/generative_strategies/SketchyGanStrategy.h)
target_link_libraries(client
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        "${TORCH_LIBRARIES}"
        )

set_property(TARGET client PROPERTY CXX_STANDARD 17)

set(MODEL_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/SketchyGenerator.pt")
set(MODEL_DEST_PATH "${CMAKE_CURRENT_BINARY_DIR}/SketchyGenerator.pt")

message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "MODEL_SOURCE_PATH: ${MODEL_SOURCE_PATH}")
message(STATUS "MODEL_DEST_PATH: ${MODEL_DEST_PATH}")

add_custom_command(
        OUTPUT ${MODEL_DEST_PATH}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MODEL_SOURCE_PATH} ${MODEL_DEST_PATH}
        DEPENDS ${MODEL_SOURCE_PATH}
)

add_custom_target(copy_model ALL DEPENDS ${MODEL_DEST_PATH})