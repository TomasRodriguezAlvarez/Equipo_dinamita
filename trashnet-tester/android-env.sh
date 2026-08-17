# Entorno para compilar el APK. El JDK y el SDK están instalados en $HOME, no en
# el sistema (no había sudo sin contraseña), así que hay que exportarlos a mano.
#
#     source android-env.sh
#
# Si te cansa hacerlo cada vez, pegá estas mismas líneas al final de ~/.bashrc.

export JAVA_HOME="$HOME/opt/jdk-21.0.12+8"
export ANDROID_HOME="$HOME/Android/Sdk"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"
