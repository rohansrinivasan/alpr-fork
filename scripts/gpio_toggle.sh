echo 216 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio216/direction

echo 50 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio50/direction

echo 232 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio232/direction

echo 15 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio15/direction

echo 17 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio17/direction

echo 79 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio79/direction

echo 14 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio14/direction

echo 194 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio194/direction

echo 16 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio16/direction

echo 13 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio13/direction


while true; do

    echo 1 > /sys/class/gpio/gpio216/value
    echo 1 > /sys/class/gpio/gpio50/value
    echo 1 > /sys/class/gpio/gpio232/value
    echo 1 > /sys/class/gpio/gpio15/value
    echo 1 > /sys/class/gpio/gpio17/value
    echo 1 > /sys/class/gpio/gpio79/value
    echo 1 > /sys/class/gpio/gpio14/value
    echo 1 > /sys/class/gpio/gpio194/value
    echo 1 > /sys/class/gpio/gpio16/value
    echo 1 > /sys/class/gpio/gpio13/value

        sleep 2

    echo 0 > /sys/class/gpio/gpio216/value
    echo 0 > /sys/class/gpio/gpio50/value
    echo 0 > /sys/class/gpio/gpio232/value
    echo 0 > /sys/class/gpio/gpio15/value
    echo 0 > /sys/class/gpio/gpio17/value
    echo 0 > /sys/class/gpio/gpio79/value
    echo 0 > /sys/class/gpio/gpio14/value
    echo 0 > /sys/class/gpio/gpio194/value
    echo 0 > /sys/class/gpio/gpio16/value
    echo 0 > /sys/class/gpio/gpio13/value

        sleep 1
done
