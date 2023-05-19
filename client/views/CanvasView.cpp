//
// Created by maneologu on 01.04.2023.
//

#include "CanvasView.h"
#include <QMouseEvent>
#include <QPainter>
#include <QDebug>

CanvasView::CanvasView(CanvasModel *canvasModel, QWidget *parent)
    : QWidget(parent)
    , _canvasModel(canvasModel)
    , _pen(Qt::black, 5, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin)
    , _image(512, 512, QImage::Format_ARGB32)
    , _shadow(512, 512, QImage::Format_ARGB32) {
    setAttribute(Qt::WA_StaticContents);
    setMouseTracking(true);
    _image.fill(Qt::transparent);
    _shadow.fill(Qt::white);
}

void CanvasView::paintEvent(QPaintEvent *event) {
    QPainter painter(this);
    painter.drawImage(event->rect(), _shadow, event->rect());
    painter.drawImage(event->rect(), _image, event->rect());
}

void CanvasView::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        //_lastPoint = event->pos();
        emit beginDraw(event->pos());
    }
}

void CanvasView::mouseMoveEvent(QMouseEvent *event) {
    if (event->buttons() & Qt::LeftButton) {
        qDebug() << "Mouse move";
        //drawLineTo(event->pos());
        emit draw(event->pos());
    }
}

void CanvasView::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        //drawLineTo(event->pos());
        emit endDraw();
        //emit imageChanged(_image);
    }
}

void CanvasView::resizeEvent(QResizeEvent *event) {
    if (width() > _image.width() || height() > _image.height()) {
        int newWidth = qMax(width(), _image.width());
        int newHeight = qMax(height(), _image.height());
        QImage newImage(QSize(newWidth, newHeight), QImage::Format_ARGB32_Premultiplied);
        newImage.fill(Qt::white);
        QPainter painter(&newImage);
        painter.drawImage(QPoint(0, 0), _image);
        _image = newImage;
    }
    QWidget::resizeEvent(event);
}

void CanvasView::drawLineTo(const QPoint &endPoint) {
    QPainter painter(&_image);
    painter.setPen(_pen);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.drawLine(_lastPoint, endPoint);
    update(QRect(_lastPoint, endPoint).normalized().adjusted(-_pen.width() / 2, -_pen.width() / 2, _pen.width() / 2, _pen.width() / 2));
    _lastPoint = endPoint;
}

void CanvasView::updateImage(const QImage &image) {
    qDebug() << "Update image";
    _image = image;
    update();
}

void CanvasView::updateShadow(const QImage &image) {
    _shadow = image;
    update();
}
