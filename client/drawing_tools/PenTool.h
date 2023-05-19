//
// Created by maneologu on 04.05.2023.
//

#ifndef CLIENT_PENTOOL_H
#define CLIENT_PENTOOL_H

#include "DrawTool.h"

class PenTool : public DrawTool {
public:
    PenTool() {
        _pen.setWidth(1);
        _pen.setColor(Qt::black);
    }

    void draw(QPainter &painter, const QPoint &fromPos, const QPoint &toPos) override {
        painter.setPen(_pen);
        painter.drawLine(fromPos, toPos);
    }

    void setWidth(int width) override {
        _pen.setWidth(width);
    }

private:
    QPen _pen;
};


#endif //CLIENT_PENTOOL_H
