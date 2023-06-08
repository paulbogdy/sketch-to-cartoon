//
// Created by maneologu on 01.04.2023.
//

#include "CanvasModel.h"
#include "../drawing_tools/PenTool.h"
#include <QDebug>
#include <utility>

CanvasModel::CanvasModel(QObject *parent) : QObject(parent)
    , _image(512, 512, QImage::Format_ARGB32)
    , _shadow(512, 512, QImage::Format_ARGB32)
    , _drawTool(std::make_unique<PenTool>()) {
    _image.fill(Qt::transparent);
    _shadow.fill(Qt::white);
}

QImage CanvasModel::image() const {
    return _image;
}

QImage CanvasModel::shadow() const {
    return _shadow;
}

void CanvasModel::setImage(const QImage &image) {
    if(_image != image) {
        if(_undoStack.size() >= MAX_UNDO_STACK_SIZE) {
            _undoStack.removeFirst();
        }
        _undoStack.push_back(_image);
        _redoStack.clear();
        _image = image;
        emit imageChanged(_image);
        emit undoChanged(true);
        emit redoChanged(false);
    }
}

void CanvasModel::setShadow(const QImage &image) {
    if(_shadow != image) {
        _shadow = image;
        qDebug() << "Shadow changed";
        emit shadowChanged(_shadow);
    }
}

void CanvasModel::undo() {
    if (!_undoStack.empty()) {
        _redoStack.push_back(_image);
        _image = _undoStack.pop();
        emit imageChanged(_image);
        emit redoChanged(true);
        if(_undoStack.empty()) {
            emit undoChanged(false);
        }
    }
}

void CanvasModel::redo() {
    if (!_redoStack.empty()) {
        _undoStack.push_back(_image);
        _image = _redoStack.pop();
        emit imageChanged(_image);
        emit undoChanged(true);
        if(_redoStack.empty()) {
            emit redoChanged(false);
        }
    }
}

void CanvasModel::beginDraw(const QPoint &point) {
    _lastPoint = point;
    if(_undoStack.size() >= MAX_UNDO_STACK_SIZE) {
        _undoStack.removeFirst();
    }
    _undoStack.push_back(_image);
    _redoStack.clear();
    emit undoChanged(true);
    emit redoChanged(false);
}

void CanvasModel::draw(const QPoint &point) {
    QPainter painter(&_image);
    _drawTool->draw(painter, _lastPoint, point);
    _lastPoint = point;
    emit imageChanged(_image);
}

void CanvasModel::endDraw() {
}

void CanvasModel::setDrawTool(std::shared_ptr<DrawTool> tool) {
    _drawTool = std::move(tool);
}

void CanvasModel::setDrawToolWidth(int width) {
    _drawTool->setWidth(width);
}

