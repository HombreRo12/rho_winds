# ρ-winds

Este código integra las ecuaciones del viento estelar de Parker y/o politrópico.  
Necesita como inputs la masa, radio, temperatura coronal y densidad en la base estelar. Las mismas se pueden proporcionar en cualquier unidad utilizando astropy, luego el código aprovecha esta misma libería para pasarla a las unidades de trabajo.

El código necesita las librerías scipy, numpy y astropy. El mismo fue desarrollado para Python 3.10.18.
El ejemplo, utiliza matplotlib para graficar.

Las ecuaciones utilizadas fueron extraídas del libro Introduction to Stellar Winds:

Lamers H. J. G. L. M., Cassinelli J. P., 1999, Introduction to Stellar Winds, Cambridge Univ. Press, Cambridge
