#ifndef _ANN_TYPE_TRAITS_H_
#define _ANN_TYPE_TRAITS_H_

// Length
template<int...Rest> struct pack_len;
template<int Head, int ...Rest>
struct pack_len<Head, Rest...> { static const int value = 1+pack_len<Rest...>::value; };
template<> struct pack_len<>   { static const int value = 0;};

// Sum
template<int...Rest> struct pack_add;
template<int Head, int ...Rest>
struct pack_add<Head, Rest...> { static const int value = Head+pack_add<Rest...>::value;};
template<> struct pack_add<>   { static const int value = 0;};

// Multiply
template<int...Rest> struct pack_prod;
template<int Head, int ...Rest>
struct pack_prod<Head, Rest...> { static const int value = Head*pack_prod<Rest...>::value;};
template<> struct pack_prod<>   { static const int value = 1;};

// NN weights size
template<int...Rest> struct ann_weights_len;
template<int Left, int Right, int ...Rest>
struct ann_weights_len<Left, Right, Rest...> { static const int value = (Left*Right) + ann_weights_len<Right, Rest...>::value;};
template<int Last> struct ann_weights_len<Last> {static const int value = 0; };

// NN last node len
template<int...Rest> struct ann_output_len;
template<int Left, int ...Rest>
struct ann_output_len<Left, Rest...> { static const int value = ann_output_len<Rest...>::value;};
template<int Last> struct ann_output_len<Last> {static const int value = Last; };

// get maximum layer length
template<int...Rest> struct max_layer_len;
template<int Left, int ...Rest>
struct max_layer_len<Left, Rest...> { static const int value = (Left) > max_layer_len<Rest...>::value?Left:max_layer_len<Rest...>::value; };
template<int Last> struct max_layer_len<Last> { static const int value = Last; };

#endif
