//
// Created by maneologu on 02.04.2023.
//

#ifndef CLIENT_SETTINGS_H
#define CLIENT_SETTINGS_H

#include <QObject>
#include "../strategies/generative_strategies/GenerativeStrategy.h"
#include <memory>
#include <mutex>

class Settings : public QObject {
Q_OBJECT
public:
    static Settings& getInstance();

    ~Settings() override;
    void setGenerativeStrategy(GenerativeStrategy* strategy);
    GenerativeStrategy* getGenerativeStrategy();

    // Delete copy constructor and assignment operator
    Settings(const Settings&) = delete;
    Settings& operator=(const Settings&) = delete;

private:
    explicit Settings(QObject* parent = nullptr);
    GenerativeStrategy* _generativeStrategy;

    static std::shared_ptr<Settings> _instance;
    static std::mutex _mutex;
};


#endif //CLIENT_SETTINGS_H
