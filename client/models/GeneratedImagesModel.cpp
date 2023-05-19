//
// Created by maneologu on 02.04.2023.
//

#include "GeneratedImagesModel.h"

GeneratedImagesModel::GeneratedImagesModel(QObject *parent) : QObject(parent) {
}

int GeneratedImagesModel::imageCount() const {
    return _images.size();
}

QImage GeneratedImagesModel::imageAt(int index) const {
    if (index >= 0 && index < _images.size()) {
        return _images.at(index);
    }
    return QImage();
}

void GeneratedImagesModel::addImage(const QImage &image) {
    _images.append(image);
    emit imagesChanged();
}

void GeneratedImagesModel::removeImage(int index) {
    if (index >= 0 && index < _images.size()) {
        _images.remove(index);
        emit imagesChanged();
    }
}

void GeneratedImagesModel::setImages(const QVector<QImage> &images) {
    _images = QVector<QImage>(images.begin(), images.end());
    emit imagesChanged();
}
