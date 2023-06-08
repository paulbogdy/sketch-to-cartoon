//
// Created by maneologu on 01.04.2023.
//

#include "CanvasView.h"
#include <QMouseEvent>
#include <QPainter>
#include <QDebug>
#include <cmath>

CanvasView::CanvasView(CanvasModel *canvasModel, QWidget *parent)
    : QWidget(parent)
    , _canvasModel(canvasModel)
    , _pen(Qt::black, 5, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin)
    , _image(512, 512, QImage::Format_ARGB32)
    , _shadow(512, 512, QImage::Format_ARGB32)
    , _background(512, 512, QImage::Format_RGB32) {
    setAttribute(Qt::WA_StaticContents);
    setMouseTracking(true);
    _image.fill(Qt::transparent);
    _shadow.fill(Qt::white);
    _background.fill(Qt::white);
}

double CanvasView::getOpacityWeight(QPoint mousePos) {
    // Center of the widget
    QPoint center(width() / 2, height() / 2);
    // Maximum distance from the center (half the diagonal of the widget)
    double maxDist = sqrt(pow(width(), 2) + pow(height(), 2)) / 2;
    // Actual distance from the center
    double dist = (mousePos - center).manhattanLength(); // use manhattanLength() for "city block" distance
    // Weight factor, ranging from 0 (mouse at center) to 1 (mouse at corner)
    double weight = dist / maxDist;
    // Optionally, use sqrt(weight) to increase the area of low opacity
    return weight;
}

void CanvasView::setOpacityCircular(QImage &image, const QPoint& center, double radius) {
    if (image.format() != QImage::Format_ARGB32 && image.format() != QImage::Format_ARGB32_Premultiplied) {
        image = image.convertToFormat(QImage::Format_ARGB32);
    }
    quint8 *data = image.bits();

    // For each pixel in the image
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            QPoint pt(x, y);
            double distance = std::sqrt((pt.x() - center.x())*(pt.x() - center.x()) +
                                        (pt.y() - center.y())*(pt.y() - center.y())); // Euclidean distance

            // Normalize the distance to the range [0, 1]
            double normalizedDistance = std::min(distance / radius, 0.95);

            // Calculate the opacity based on the distance
            double opacity = 1.0 - normalizedDistance; // Opacity decreases with distance
            data[3] = data[3] * opacity; // Apply the opacity to the alpha channel

            data += 4; // Move to the next pixel
        }
    }
}


void CanvasView::paintEvent(QPaintEvent *event) {
    QPainter painter(this);

    // Paint background
    painter.drawImage(event->rect(), _background, event->rect());

    if(_showShadow) {
        // Get the cursor position relative to the widget
        QPoint cursorPos = mapFromGlobal(QCursor::pos());

        // Convert the cursor position to the image's coordinate space
        QPoint imagePos(cursorPos.x() * _shadow.width() / width(),
                        cursorPos.y() * _shadow.height() / height());

        QImage shadowWithOpacity = _shadow.copy(); // Make a copy if you need the original shadow later
        setOpacityCircular(shadowWithOpacity, imagePos, 200); // Set radius as per your requirement
        painter.drawImage(event->rect(), shadowWithOpacity, event->rect());
    }

    // Paint image
    painter.drawImage(event->rect(), _image, event->rect());
}




void CanvasView::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        emit beginDraw(event->pos());
    }
}

void CanvasView::mouseMoveEvent(QMouseEvent *event) {
    if (event->buttons() & Qt::LeftButton) {
        emit draw(event->pos());
    } else if (_showShadow){
        update();
    }
}

void CanvasView::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        emit endDraw();
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
    _image = image;
    update();
}

void CanvasView::updateShadow(const QImage &image) {
    _shadow = image;
    if (_showShadow) {
        update();
    }
}

void CanvasView::updateShadowVisibility(int state) {
    bool showShadow = false;
    if(state == Qt::Checked) {
        showShadow = true;
    }
    if(showShadow != _showShadow) {
        _showShadow = showShadow;
        update();
    }
}
