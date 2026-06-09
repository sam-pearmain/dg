use crate::{
    quadrule_impl,
    quadrules::quadrature::{GaussLegendre, GaussLegendreLobatto, ShapeQuadrature},
    shapes::Line,
};

/// A Gauss-Legendre-Lobatto line
pub type GaussLegendreLobattoLine<F, const D: usize, const N: usize> =
    ShapeQuadrature<F, Line<F>, GaussLegendreLobatto, D, N>;

/// A Gauss-Legendre line
pub type GaussLegendreLine<F, const D: usize, const N: usize> =
    ShapeQuadrature<F, Line<F>, GaussLegendre, D, N>;

// gauss-legendre-lobatto line aliases
pub type GaussLegendreLobattoLineD1N2<F> = GaussLegendreLobattoLine<F, 1, 2>;
pub type GaussLegendreLobattoLineD3N3<F> = GaussLegendreLobattoLine<F, 3, 3>;
pub type GaussLegendreLobattoLineD5N4<F> = GaussLegendreLobattoLine<F, 5, 4>;
pub type GaussLegendreLobattoLineD7N5<F> = GaussLegendreLobattoLine<F, 7, 5>;
pub type GaussLegendreLobattoLineD9N6<F> = GaussLegendreLobattoLine<F, 9, 6>;
pub type GaussLegendreLobattoLineD11N7<F> = GaussLegendreLobattoLine<F, 11, 7>;
pub type GaussLegendreLobattoLineD13N8<F> = GaussLegendreLobattoLine<F, 13, 8>;
pub type GaussLegendreLobattoLineD15N9<F> = GaussLegendreLobattoLine<F, 15, 9>;
pub type GaussLegendreLobattoLineD17N10<F> = GaussLegendreLobattoLine<F, 17, 10>;
pub type GaussLegendreLobattoLineD19N11<F> = GaussLegendreLobattoLine<F, 17, 11>;
pub type GaussLegendreLobattoLineD21N12<F> = GaussLegendreLobattoLine<F, 21, 12>;
pub type GaussLegendreLobattoLineD23N13<F> = GaussLegendreLobattoLine<F, 23, 13>;
pub type GaussLegendreLobattoLineD25N14<F> = GaussLegendreLobattoLine<F, 25, 14>;
pub type GaussLegendreLobattoLineD27N15<F> = GaussLegendreLobattoLine<F, 27, 15>;
pub type GaussLegendreLobattoLineD29N16<F> = GaussLegendreLobattoLine<F, 29, 16>;
pub type GaussLegendreLobattoLineD31N17<F> = GaussLegendreLobattoLine<F, 31, 17>;
pub type GaussLegendreLobattoLineD33N18<F> = GaussLegendreLobattoLine<F, 33, 18>;
pub type GaussLegendreLobattoLineD35N19<F> = GaussLegendreLobattoLine<F, 35, 19>;
pub type GaussLegendreLobattoLineD37N20<F> = GaussLegendreLobattoLine<F, 37, 20>;

// gauss-legendre line aliases
pub type GaussLegendreLineD1N1<F> = GaussLegendreLine<F, 1, 1>;
pub type GaussLegendreLineD3N2<F> = GaussLegendreLine<F, 3, 2>;
pub type GaussLegendreLineD5N3<F> = GaussLegendreLine<F, 5, 3>;
pub type GaussLegendreLineD7N4<F> = GaussLegendreLine<F, 7, 4>;
pub type GaussLegendreLineD9N5<F> = GaussLegendreLine<F, 9, 5>;
pub type GaussLegendreLineD11N6<F> = GaussLegendreLine<F, 11, 6>;
pub type GaussLegendreLineD13N7<F> = GaussLegendreLine<F, 13, 7>;
pub type GaussLegendreLineD15N8<F> = GaussLegendreLine<F, 15, 8>;
pub type GaussLegendreLineD17N9<F> = GaussLegendreLine<F, 17, 9>;
pub type GaussLegendreLineD19N10<F> = GaussLegendreLine<F, 19, 10>;
pub type GaussLegendreLineD21N11<F> = GaussLegendreLine<F, 21, 11>;
pub type GaussLegendreLineD23N12<F> = GaussLegendreLine<F, 23, 12>;
pub type GaussLegendreLineD25N13<F> = GaussLegendreLine<F, 25, 13>;
pub type GaussLegendreLineD27N14<F> = GaussLegendreLine<F, 27, 14>;
pub type GaussLegendreLineD29N15<F> = GaussLegendreLine<F, 29, 15>;
pub type GaussLegendreLineD31N16<F> = GaussLegendreLine<F, 31, 16>;
pub type GaussLegendreLineD33N17<F> = GaussLegendreLine<F, 33, 17>;
pub type GaussLegendreLineD35N18<F> = GaussLegendreLine<F, 35, 18>;
pub type GaussLegendreLineD37N19<F> = GaussLegendreLine<F, 37, 19>;
pub type GaussLegendreLineD39N20<F> = GaussLegendreLine<F, 39, 20>;

quadrule_impl!(
    GaussLegendreLobattoLineD1N2,
    points:
    [
        [-1.0],
        [1.0]
    ],
    weights:
    [
        1.0,
        1.0
    ]
);

quadrule_impl!(
    GaussLegendreLobattoLineD3N3,
    points:
    [
        [-1.0],
        [0.0],
        [1.0]
    ],
    weights:
    [
        0.333333333333333333333333333333,
        1.333333333333333333333333333333,
        0.333333333333333333333333333333,
    ]
);

quadrule_impl!(
    GaussLegendreLobattoLineD5N4,
    points:
    [
        [-1.0],
        [-0.447213595499957939281834733746],
        [0.447213595499957939281834733746],
        [1.0],
    ],
    weights: [
        0.166666666666666666666666666667,
        0.833333333333333333333333333333,
        0.833333333333333333333333333333,
        0.166666666666666666666666666667,
    ]
);

quadrule_impl!(
    GaussLegendreLobattoLineD7N5,
    points:
    [
        [-1.0],
        [-0.654653670707977143798292456247],
        [0.0],
        [0.654653670707977143798292456247],
        [1.0]
    ],
    weights:
    [
        0.1,
        0.544444444444444444444444444444,
        0.711111111111111111111111111111,
        0.544444444444444444444444444444,
        0.1,
    ]
);

quadrule_impl!(
    GaussLegendreLobattoLineD9N6, 
    points:
    [
        [-1.0],
        [-0.765055323929464692851002973959],
        [-0.285231516480645096314150994041],
        [0.285231516480645096314150994041],
        [0.765055323929464692851002973959],
        [1.0],
    ], 
    weights:
    [
        0.0666666666666666666666666666667,
        0.378474956297846980316612808212,
        0.554858377035486353016720525121,
        0.554858377035486353016720525121,
        0.378474956297846980316612808212,
        0.0666666666666666666666666666667,
    ]
);

quadrule_impl!(
    GaussLegendreLobattoLineD11N7, 
    points:
    [
        [-1.0],
        [-0.830223896278566929872032213967],
        [-0.468848793470714213803771881909],
        [0.0],
        [0.468848793470714213803771881909],
        [0.830223896278566929872032213967],
        [1.0],
    ], 
    weights:
    [
        0.047619047619047619047619047619,
        0.27682604736156594801070040629,
        0.431745381209862623417871022281,
        0.487619047619047619047619047619,
        0.431745381209862623417871022281,
        0.27682604736156594801070040629,
        0.047619047619047619047619047619,
    ]
);

quadrule_impl!(
    GaussLegendreLobattoLineD13N8, 
    points:
    [
        [-1.0],
        [-0.871740148509606615337445761221],
        [-0.591700181433142302144510731398],
        [-0.209299217902478868768657260345],
        [0.209299217902478868768657260345],
        [0.591700181433142302144510731398],
        [0.871740148509606615337445761221],
        [1.0],
    ], 
    weights:
    [
        0.0357142857142857142857142857143,
        0.210704227143506039382992065776,
        0.341122692483504364764240677108,
        0.412458794658703881567052971402,
        0.412458794658703881567052971402,
        0.341122692483504364764240677108,
        0.210704227143506039382992065776,
        0.0357142857142857142857142857143,
    ]
);

quadrule_impl!(
    GaussLegendreLobattoLineD15N9, 
    points: 
    [
        [-1.0],
        [-0.899757995411460157312345244418],
        [-0.677186279510737753445885427091],
        [-0.363117463826178158710752068709],
        [0.0],
        [0.363117463826178158710752068709],
        [0.677186279510737753445885427091],
        [0.899757995411460157312345244418],
        [1.0],
    ], 
    weights:
    [
        0.0277777777777777777777777777778,
        0.165495361560805525046339720029,
        0.274538712500161735280705618579,
        0.34642851097304634511513153214,
        0.371519274376417233560090702948,
        0.34642851097304634511513153214,
        0.274538712500161735280705618579,
        0.165495361560805525046339720029,
        0.0277777777777777777777777777778,
    ]
);

quadrule_impl!(
    GaussLegendreLobattoLineD17N10, 
    points: 
    [
        [-1.0],
        [-0.919533908166458813828932660822],
        [-0.73877386510550507500310617486],
        [-0.477924949810444495661175092731],
        [-0.165278957666387024626219765958],
        [0.165278957666387024626219765958],
        [0.477924949810444495661175092731],
        [0.73877386510550507500310617486],
        [0.919533908166458813828932660822],
        [1.0],
    ], 
    weights:
    [
        0.0222222222222222222222222222222,
        0.133305990851070111126227170755,
        0.224889342063126452119457821731,
        0.292042683679683757875582257374,
        0.327539761183897456656510527917,
        0.327539761183897456656510527917,
        0.292042683679683757875582257374,
        0.224889342063126452119457821731,
        0.133305990851070111126227170755,
        0.0222222222222222222222222222222,
    ]
);

quadrule_impl!(
    GaussLegendreLobattoLineD19N11, 
    points:
    [
        [-1.0],
        [-0.934001430408059134332274136099],
        [-0.784483473663144418622417816108],
        [-0.565235326996205006470963969478],
        [-0.295758135586939391431911515559],
        [0.0],
        [0.295758135586939391431911515559],
        [0.565235326996205006470963969478],
        [0.784483473663144418622417816108],
        [0.934001430408059134332274136099],
        [1.0],
    ], 
    weights:
    [
        0.0181818181818181818181818181818,
        0.10961227326699486446140344958,
        0.187169881780305204108141521899,
        0.248048104264028314040084866422,
        0.286879124779008088679222403332,
        0.30021759545569069378593188117,
        0.286879124779008088679222403332,
        0.248048104264028314040084866422,
        0.187169881780305204108141521899,
        0.10961227326699486446140344958,
        0.0181818181818181818181818181818,
    ] 
);

quadrule_impl!(
    GaussLegendreLobattoLineD21N12, 
    points: 
    [
        [-1.0],
        [-0.944899272222882223407580138303],
        [-0.819279321644006678348641581717],
        [-0.632876153031860677662404854444],
        [-0.399530940965348932264349791567],
        [-0.13655293285492755486406185574],
        [0.13655293285492755486406185574],
        [0.399530940965348932264349791567],
        [0.632876153031860677662404854444],
        [0.819279321644006678348641581717],
        [0.944899272222882223407580138303],
        [1.0],
    ], 
    weights:
    [
        0.0151515151515151515151515151515,
        0.0916845174131961306683425941341,
        0.1579747055643701151646710627,
        0.212508417761021145358302077367,
        0.251275603199201280293244412148,
        0.2714052409106961770002883385,
        0.2714052409106961770002883385,
        0.251275603199201280293244412148,
        0.212508417761021145358302077367,
        0.1579747055643701151646710627,
        0.0916845174131961306683425941341,
        0.0151515151515151515151515151515,
    ]
);
