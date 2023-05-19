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
find_package(Qt5 COMPONENTS Network REQUIRED)

add_executable(client main.cpp UI/mainwindow.cpp UI/mainwindow.h UI/SettingsWindow.cpp UI/SettingsWindow.h models/CanvasModel.cpp models/CanvasModel.h views/CanvasView.cpp views/CanvasView.h models/GeneratedImagesModel.cpp models/GeneratedImagesModel.h views/GeneratedImagesView.cpp views/GeneratedImagesView.h controllers/MainController.cpp controllers/MainController.h settings/Settings.cpp settings/Settings.h strategies/generative_strategies/GenerativeStrategy.cpp strategies/generative_strategies/GenerativeStrategy.h strategies/generative_strategies/DefaultStrategy.cpp strategies/generative_strategies/DefaultStrategy.h strategies/generative_strategies/SketchyGanStrategy.cpp strategies/generative_strategies/SketchyGanStrategy.h strategies/generative_strategies/GanInversionStrategy.cpp strategies/generative_strategies/GanInversionStrategy.h controllers/CanvasController.cpp controllers/CanvasController.h controllers/CanvasController.h controllers/GeneratedImagesController.cpp controllers/GeneratedImagesController.h models/MessageBus.cpp models/MessageBus.h views/BottomBar.cpp views/BottomBar.h strategies/generative_strategies/GenerativeStrategyFactory.cpp strategies/generative_strategies/GenerativeStrategyFactory.h views/FullSizeImageWidget.cpp views/FullSizeImageWidget.h drawing_tools/DrawTool.cpp drawing_tools/DrawTool.h drawing_tools/PenTool.cpp drawing_tools/PenTool.h drawing_tools/EraserTool.cpp drawing_tools/EraserTool.h)

target_link_libraries(client
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::Network
        "${TORCH_LIBRARIES}"
        )

set_property(TARGET client PROPERTY CXX_STANDARD 17)

set(MODEL_SOURCE_PATH "${CMAKE_SOURCE_DIR}/SketchyGenerator.pt")
set(MODEL_DEST_PATH "${CMAKE_BINARY_DIR}/SketchyGenerator.pt")

message(STATUS "Copying model from ${MODEL_SOURCE_PATH} to ${MODEL_DEST_PATH}")

add_custom_command(
        OUTPUT ${MODEL_DEST_PATH}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MODEL_SOURCE_PATH} ${MODEL_DEST_PATH}
        DEPENDS ${MODEL_SOURCE_PATH}
)

add_custom_target(copy_model ALL DEPENDS ${MODEL_DEST_PATH})

# Copy the resources folder to the output directory only when its contents change
file(GLOB_RECURSE RESOURCE_FILES "${CMAKE_SOURCE_DIR}/resources/*")
add_custom_target(copy_resources ALL)
foreach(RESOURCE_FILE ${RESOURCE_FILES})
    file(RELATIVE_PATH RELATIVE_PATH "${CMAKE_SOURCE_DIR}/resources" ${RESOURCE_FILE})
    message(STATUS "Copying resource ${RESOURCE_FILE} to ${RELATIVE_PATH}")
    set(RESOURCE_DEST_PATH "${CMAKE_BINARY_DIR}/resources/${RELATIVE_PATH}")
    add_custom_command(
            TARGET copy_resources POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/resources
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RESOURCE_FILE} ${RESOURCE_DEST_PATH}
            DEPENDS ${RESOURCE_FILE}
            COMMENT "Copying ${RESOURCE_FILE} to ${RESOURCE_DEST_PATH}"
    )
endforeach()

