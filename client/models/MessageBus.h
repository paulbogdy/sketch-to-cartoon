//
// Created by maneologu on 03.05.2023.
//

#ifndef CLIENT_MESSAGEBUS_H
#define CLIENT_MESSAGEBUS_H

#include <QObject>

class MessageBus : public QObject {
Q_OBJECT
public:
    static MessageBus &getInstance() {
        static MessageBus instance;
        return instance;
    }
    void publishMessage(const QString &message, bool isLoading) {
        emit messagePublished(message, isLoading);
    }
signals:
    void messagePublished(const QString &message, bool isLoading);
private:
    MessageBus() = default;
    ~MessageBus() = default;
    Q_DISABLE_COPY(MessageBus)
};


#endif //CLIENT_MESSAGEBUS_H
