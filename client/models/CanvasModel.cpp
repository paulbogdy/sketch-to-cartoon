//
// Created by maneologu on 01.04.2023.
//

#include "CanvasModel.h"

CanvasModel::CanvasModel(QObject *parent) : QObject(parent) {

}

QImage CanvasModel::image() const {
    return _image;
}

void CanvasModel::setImage(const QImage &image) {
    if(_image != image) {
        _image = image;
        emit imageChanged(_image);
    }
}
