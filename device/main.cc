#include <iostream>

#include "sound_assist.h"

int main() {
    std::cout << "Starting main" << std::endl;

    SoundAssist sound_assist;
    sound_assist.run();

    std::cout << "Ending main" << std::endl;
}
