//
// Created by maneologu on 03.04.2023.
//

#include "SketchyGanStrategy.h"
#include <QImage>
#include <QVector>
#include <QColor>
#include <filesystem>
#include <iostream>

QVector<QImage> SketchyGanStrategy::generateFromSketch(QImage sketch) {
    // Convert the input sketch image to grayscale and resize it to (64, 64)
    QImage gray_image = sketch.scaled(64, 64, Qt::KeepAspectRatio, Qt::SmoothTransformation).convertToFormat(QImage::Format_Grayscale8);

    // Convert the grayscale image to a PyTorch tensor and normalize it
    torch::Tensor gray_image_tensor = torch::zeros({64, 64});
    for (int y = 0; y < gray_image.height(); y++) {
        for (int x = 0; x < gray_image.width(); x++) {
            gray_image_tensor[y][x] = QColor(gray_image.pixel(x, y)).value() / 255.0;
        }
    }

    // Repeat the grayscale image tensor 8 times along the batch dimension
    torch::Tensor sketch_batch = gray_image_tensor.unsqueeze(0).repeat({8, 1, 1, 1});

    // Generate random noise of shape (8, 1, 64, 64)
    torch::Tensor noise = torch::randn({8, 1, 64, 64});

    noise = noise.to(_device);
    sketch_batch = sketch_batch.to(_device);

    // Pass the random noise and repeated sketch image tensor to the generator
    torch::Tensor generated_images = _generator.forward({noise, sketch_batch}).toTensor();

    // Convert the generated images tensor back to QVector<QImage>
    QVector<QImage> generated_qimages;
    for (int i = 0; i < generated_images.size(0); i++) {
        QImage generated_qimage(64, 64, QImage::Format_RGB32);
        auto img_data = generated_images[i].permute({1, 2, 0}).mul(255.0).clamp(0, 255).to(torch::kU8).cpu().contiguous();
        std::memcpy(generated_qimage.bits(), img_data.data_ptr(), img_data.numel() * sizeof(torch::kU8));
        generated_qimages.push_back(generated_qimage.scaled(128, 128, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    return generated_qimages;
}


SketchyGanStrategy::~SketchyGanStrategy() {

}

SketchyGanStrategy::SketchyGanStrategy(): _device(torch::kCPU) {
    try {
        if (!torch::cuda::is_available()) {
            std::cerr << "CUDA is not available. Using CPU instead." << std::endl;
            _device = torch::kCPU;
        }
        _generator = torch::jit::load("SketchyGenerator.pt");
        _generator.to(_device);
    } catch (const c10::Error &e) {
        // Handle error during model loading
        std::cerr << "Error loading the model\n" << e.msg() << std::endl;
    }
}
