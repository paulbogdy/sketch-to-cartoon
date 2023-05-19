//
// Created by maneologu on 01.04.2023.
//

#ifndef CLIENT_CANVASVIEW_H
#define CLIENT_CANVASVIEW_H


#include <QWidget>
#include <QPoint>
#include <QPen>
#include "../models/CanvasModel.h"


class CanvasView : public QWidget {
Q_OBJECT

public:
    explicit CanvasView(CanvasModel* canvasModel, QWidget* parent = nullptr);

public slots:
    void updateImage(const QImage& image);
    void updateShadow(const QImage& image);

signals:
    void imageChanged(const QImage& image);
    void undoRequest();
    void redoRequest();
    void beginDraw(const QPoint& point);
    void draw(const QPoint& point);
    void endDraw();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void drawLineTo(const QPoint& endPoint);

    CanvasModel* _canvasModel;
    QImage _image;
    QImage _shadow;
    QPoint _lastPoint;
    QPen _pen;
};



#endif //CLIENT_CANVASVIEW_H
