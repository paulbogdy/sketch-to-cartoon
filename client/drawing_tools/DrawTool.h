//
// Created by maneologu on 04.05.2023.
//

#ifndef CLIENT_DRAWTOOL_H
#define CLIENT_DRAWTOOL_H

#include <QPainter>

class DrawTool {
public:
    virtual ~DrawTool() = default;
    virtual void draw(QPainter &painter, const QPoint &fromPos, const QPoint &toPos) = 0;
    virtual void setWidth(int width) = 0;
};


#endif //CLIENT_DRAWTOOL_H
