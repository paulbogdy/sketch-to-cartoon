//
// Created by maneologu on 04.05.2023.
//

#ifndef CLIENT_ERASERTOOL_H
#define CLIENT_ERASERTOOL_H


#include "DrawTool.h"

class EraserTool : public DrawTool {
public:
    EraserTool() {
        _eraser.setWidth(10); // Eraser is wider by default
        _eraser.setBrush(QBrush(Qt::transparent)); // The eraser color is transparent
        _eraser.setCapStyle(Qt::RoundCap);
        _eraser.setJoinStyle(Qt::RoundJoin);
    }

    void draw(QPainter &painter, const QPoint &fromPos, const QPoint &toPos) override {
        QPainter::CompositionMode originalMode = painter.compositionMode();
        painter.setCompositionMode(QPainter::CompositionMode_Clear);
        painter.setPen(_eraser);
        painter.drawLine(fromPos, toPos);
        painter.setCompositionMode(originalMode);
    }

    void setWidth(int width) override {
        _eraser.setWidth(width * 5);
    }

private:
    QPen _eraser;
};



#endif //CLIENT_ERASERTOOL_H
