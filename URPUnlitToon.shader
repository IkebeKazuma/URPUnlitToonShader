Shader "Custom/CustomToonURP" {

    // プロパティ
    Properties {
        _MainColor("Main Color", Color) = (1, 1, 1, 1)
        _MainTex ("Main Tex", 2D) = "white" {}

        _NormalMap("Normal Map", 2D) = "bump" {}
        _NormalMapIntensity("Normal Map Intensity", Range(0, 8)) = 1

        _AmbientColor("Ambient Color", Color) = (1, 1, 1, 1)

        [HDR] _SpecularColor("Specular Color", Color) = (0.9,0.9,0.9,1)
        _Glossiness("Glossiness", Float) = 32

        [Toggle(USE_OUTLINE)] _UseOutline("Use Outline", int) = 1
        _OutlineWidth ("Outline width", Range (0.005, 0.03)) = 0.01
        [HDR] _OutlineColor ("Outline Color", Color) = (0,0,0,1)
        [Toggle(USE_VERTEX_EXPANSION)] _UseVertexExpansion("Use vertex for Outline", int) = 0

        [HDR] _RimColor("Rim Color", Color) = (1,1,1,1)
        _RimAmount("Rim Amount", Range(0, 1)) = 0.716
        _RimThreshold("Rim Threshold", Range(0, 1)) = 0.1
        
        _DitherTex ("Dither Pattern (R)", 2D) = "white" {}
        _Alpha ("Alpha", Range(0.0, 1.0)) = 1.0
    }

    SubShader {
        Tags { 
            "RenderType" = "Opaque"
            "RenderPipeline" = "UniversalPipeline"
            "IgnoreProjector" = "True"
            "Queue" = "Geometry"
        }

        LOD 100

        HLSLINCLUDE

            // 関数指定
            #pragma vertex vert
            #pragma fragment frag
            // フォグ
            #pragma multi_compile_fog
            
            // #include参照
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes {
                float4 positionOS   : POSITION;
                float3 normal       : NORMAL;
				float2 uv           : TEXCOORD0;
                float4 shadowCoord  : TEXCOORD3;
                float4 tangentOS  : TANGENT;
                
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings {
                float4 positionHCS  : SV_POSITION;
                float3 positionWS   : TEXCOORD0;
				float2 uv           : TEXCOORD1;
                float4 screenPos    : TEXCOORD2;
                float4 shadowCoord  : TEXCOORD3;
                float  fogFactor    : TEXCOORD4;
                float3 viewDir      : TEXCOORD5;
                float3 normal       : NORMAL;
                float3 normalWS     : NORMAL_WS;
                float4 tangentWS    : TANGENT_WS;
                
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            TEXTURE2D(_DitherTex);
			SAMPLER(sampler_DitherTex);
            half _Alpha;

        ENDHLSL

        Pass {
			Name "CustomToonURP"

            Tags {
                "LightMode" = "UniversalForward"
            }
            
            HLSLPROGRAM

            // URP shadow keywords
            #if UNITY_VERSION >= 202120
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE
#else
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS_CASCADE
#endif
            #pragma multi_compile _ _ADDITIONAL_LIGHTS
            #pragma multi_compile _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile _ _SHADOWS_SOFT

            // GPU Instancing
            #pragma multi_compile_instancing

            // #include参照
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"

            CBUFFER_START(UnityPerMaterial)
            float4 _MainTex_ST;
            float4 _NormalMap_ST;
            CBUFFER_END

			TEXTURE2D(_MainTex);
			SAMPLER(sampler_MainTex);
            
			TEXTURE2D(_NormalMap);
			SAMPLER(sampler_NormalMap);
            float _NormalMapIntensity;

            float4 _MainColor;

            float4 _AmbientColor;

            float _Glossiness;
            float4 _SpecularColor;

            float4 _RimColor;
            float _RimAmount;
            float _RimThreshold;
            
            // 頂点シェーダー
            Varyings vert(Attributes IN) {
                Varyings OUT;
                
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_TRANSFER_INSTANCE_ID(IN, OUT);
                
                // TransformObjectToHClip()で頂点位置をオブジェクト空間からクリップスペースへ変換
                OUT.positionHCS = TransformObjectToHClip(IN.positionOS.xyz);
				// OUT.uv = IN.uv;
                OUT.uv = TRANSFORM_TEX(IN.uv, _MainTex);
                OUT.normal = IN.normal;
                // TransformObjectToWorldNormal() : ノーマルをオブジェクト空間からワールド空間へ変換
                OUT.normalWS = TransformObjectToWorldNormal(IN.normal);
                // TransformObjectToWorld() : 位置情報をオブジェクト空間からワールド空間へ変換
                OUT.positionWS = TransformObjectToWorld(IN.positionOS.xyz);

                OUT.viewDir = GetWorldSpaceViewDir(OUT.positionWS);
                
                OUT.fogFactor = ComputeFogFactor(IN.positionOS.z);

                // ComputeScreenPos()
                OUT.screenPos = ComputeScreenPos(OUT.positionHCS);

                //VertexPositionInputs vertexInput = GetVertexPositionInputs(IN.positionOS.xyz);
                //OUT.shadowCoord = GetShadowCoord(vertexInput);

                real sign = IN.tangentOS.w * GetOddNegativeScale();

                OUT.tangentWS  = real4(GetVertexNormalInputs(IN.normal, IN.tangentOS).tangentWS, sign);

                return OUT;
            }

            // パラメータの構造体
            struct LightingParams{
                float3 albedo;
                float3 ambientColor;
                float3 worldPos;
                float3 normalWS;
                half3 viewDir;
                float4 shadowCoord;
                float rawShadow;

                // Baked lighting
                float3 bakedGI;
                float4 shadowMask;
                float fogFactor;
            };

            float GetRawShadow(LightingParams p, Light light){
                // 法線とライト方向の内積
                float NdotL = saturate(dot(p.normalWS, light.direction));
                float diffuse = saturate(smoothstep(0.005, 0.01, NdotL));

                // 影の減衰
                float attenuation = saturate(smoothstep(0.005, 0.01, light.distanceAttenuation * light.shadowAttenuation));
                
                float rawShadow = diffuse * attenuation;

                return rawShadow;
            }

            float3 CustomLightHandling(LightingParams p, Light light, float3 ambientColor) {

                float rawShadow = GetRawShadow(p, light);

                // 影に色付け
                float3 shadow = lerp(ambientColor, 1, rawShadow);
                
                float3 radiance = light.color * shadow;

                // スペキュラ
                half3 halfVector = normalize(light.direction + p.viewDir);
                half NdotH = saturate(dot(p.normalWS, halfVector));
                half specularIntensity = pow(NdotH, _Glossiness * _Glossiness);
                half specularIntensitySmooth = smoothstep(0.005, 0.01, specularIntensity);
                half specular = (specularIntensitySmooth * _SpecularColor) * rawShadow;

                // リム
                float rimDot = saturate(1 - dot(p.viewDir, p.normalWS));
                float rimIntensity = rimDot * pow(NdotH, _RimThreshold);
                rimIntensity = smoothstep(_RimAmount - 0.01, _RimAmount + 0.01, rimIntensity);
                float3 rim = rimIntensity * _RimColor * shadow;

                float3 color = (p.albedo + specular + rim) * radiance;

                return color;
            }

            // 実際にカラーやライティングの計算を行い最終結果を返す
            // 引数でパラメータを受け取る
            float3 CalcCustomLighting(LightingParams p) {
                // MainLight取得
                Light mainLight = GetMainLight(p.shadowCoord, p.worldPos, 1);

                float3 color = 0;
                float tmpShadow = 0;
                // Shade the main light
                tmpShadow = GetRawShadow(p, mainLight);
                color += CustomLightHandling(p, mainLight, p.ambientColor);

                // Additional lights
                #ifdef _ADDITIONAL_LIGHTS
                // Shade additional cone and point lights. Functions in URP/ShaderLibrary/Lighting.hlsl
                uint numAdditionalLights = GetAdditionalLightsCount();
                for (uint lightI = 0; lightI < numAdditionalLights; lightI++) {
                    Light light = GetAdditionalLight(lightI, p.worldPos, 1);
                    tmpShadow = GetRawShadow(p, light);
                    color += CustomLightHandling(p, light, 0);
                }
                #endif
                
                color = MixFog(color, p.fogFactor);
                
                return color;
            }

            // フラグメントシェーダー
            float4 frag(Varyings IN) : SV_Target {
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                float4 positionCS = TransformWorldToHClip(IN.positionWS);

                // 深度バッファのサンプリング用の UV 座標を算出する
                // ピクセル位置をレンダーターゲットの解像度（_ScaledScreenParams）で除算
                float2 depthUV = IN.positionHCS.xy / _ScaledScreenParams.xy;

                // カメラ深度テクスチャから深度をサンプリング
                #if UNITY_REVERSED_Z
                    half depth = SampleSceneDepth(depthUV);
                #else
                    // Z を OpenGL の NDC ([-1, 1]) に一致するよう調整
                    half depth = lerp(UNITY_NEAR_CLIP_VALUE, 1, SampleSceneDepth(depthUV));
                #endif
                
                // スクリーン座標
                float2 screenPos = IN.screenPos.xy / IN.screenPos.w;

                LightingParams p;
                p.albedo = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, IN.uv) * _MainColor;
                p.worldPos = IN.positionWS;

                #if SHADOWS_SCREEN
                    p.shadowCoord = ComputeScreenPos(positionCS);
                #else
                    p.shadowCoord = TransformWorldToShadowCoord(p.worldPos);
                #endif

                //p.shadowCoord = IN.shadowCoord;
                p.ambientColor = _AmbientColor;
                p.normalWS = normalize(IN.normalWS);
                p.viewDir = normalize(IN.viewDir);
                p.fogFactor = ComputeFogFactor(positionCS.z);

                float3 normalTS = UnpackNormalScale(SAMPLE_TEXTURE2D(_NormalMap, sampler_NormalMap, IN.uv), _NormalMapIntensity);
                real sgn = IN.tangentWS.w;      // should be either +1 or -1
                real3 bitangent = sgn * cross(IN.normalWS.xyz, IN.tangentWS.xyz);
                real3 normalWS = mul(normalTS, real3x3(IN.tangentWS.xyz, bitangent.xyz, IN.normalWS.xyz));
                p.normalWS = normalize(normalWS);

                // ファークリップ面の付近の色を黒に設定
                #if UNITY_REVERSED_Z
                    // D3D などの REVERSED_Z があるプラットフォームの場合
                    if(depth < 0.0001)
                        return half4(0,0,0,1); 
                #else
                    // OpenGL などの REVERSED_Z がないプラットフォームの場合
                    if(depth > 0.9999)
                        return half4(0,0,0,1);
                #endif

                // ディザリングテクスチャ用のUVを作成
                float2 ditherUV = screenPos * ( _ScreenParams.xy / 4 );

                float dither = SAMPLE_TEXTURE2D(_DitherTex, sampler_DitherTex, ditherUV).r;
                clip(_Alpha - dither);
                
                return float4(CalcCustomLighting(p), 1);
            }
            ENDHLSL
        }

        //アウトライン描画
        Pass {
            Name "Outline"

            Cull Front

            HLSLPROGRAM
            
            #pragma shader_feature USE_VERTEX_EXPANSION
            #pragma shader_feature USE_OUTLINE

            half _OutlineWidth;
            half4 _OutlineColor;

            half2 TransformViewToProjection (half2 v) {
                return mul((half2x2)UNITY_MATRIX_P, v);
            }

            //頂点シェーダー
            Varyings vert(Attributes IN) {
                Varyings OUT;
                
                OUT.positionHCS = TransformObjectToHClip(IN.positionOS.xyz);

                half3 n = 0;

                #ifdef USE_VERTEX_EXPANSION //モデルの頂点方向に拡大するパターン

                //モデルの原点からみた各頂点の位置ベクトルを計算
                half3 dir = normalize(IN.positionOS.xyz);
                //UNITY_MATRIX_IT_MVはモデルビュー行列の逆行列の転置行列
                //各頂点の位置ベクトルをモデル座標系からビュー座標系に変換し正規化
                n = normalize(mul((half3x3)UNITY_MATRIX_IT_MV, dir));
                
                #else //モデルの法線方向に拡大するパターン

                //法線をモデル座標系からビュー座標系に変換し正規化
                n = normalize(mul((half3x3)UNITY_MATRIX_IT_MV, IN.normal));

                #endif

                //ビュー座標系に変換した法線を投影座標系に変換　
                //アウトラインとして描画予定であるピクセルのXY方向のオフセット
                half2 offset = TransformViewToProjection(n.xy);
                OUT.positionHCS.xy += offset * _OutlineWidth / unity_CameraProjection._m11;

                OUT.screenPos = ComputeScreenPos(OUT.positionHCS);

                OUT.positionWS = TransformObjectToWorld(IN.positionOS.xyz);
                return OUT;
            }

            //フラグメントシェーダー
            half4 frag(Varyings IN) : SV_Target {
                #ifndef USE_OUTLINE
                    discard;
                #endif

                half4 positionCS = TransformWorldToHClip(IN.positionWS);
                half fogFactor = ComputeFogFactor(positionCS.z);
                
                half3 color =MixFog(_OutlineColor.rgb, fogFactor);
                
                // ディザリングテクスチャ用のUVを作成                
                half2 screenPos = IN.screenPos.xy / IN.screenPos.w;
                half2 ditherUV = screenPos * (_ScreenParams.xy / 4);

                half dither = SAMPLE_TEXTURE2D(_DitherTex, sampler_DitherTex, ditherUV).r;
                clip(_Alpha - dither);

                return half4(color, 1);
            }

            ENDHLSL
        }

        //UsePass "Universal Render Pipeline/Lit/ShadowCaster"
        Pass {
            Name "ShadowCaster"

            Tags { "LightMode"="ShadowCaster" }

            HLSLPROGRAM

            #pragma multi_compile_instancing
            
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            // ShadowsCasterPass.hlsl に定義されているグローバルな変数
            float3 _LightDirection;

            float3 _AmbientColor;
            
            // struct Attributes {
            //     float4 positionOS : POSITION;
            //     float3 normal : NORMAL;
            //     UNITY_VERTEX_INPUT_INSTANCE_ID
            // };

            // struct Varyings {
            //     float4 posisionHCS : SV_POSITION;
            // };

            Varyings vert(Attributes IN) {
                UNITY_SETUP_INSTANCE_ID(IN);
                Varyings OUT;
                // ShadowsCasterPass.hlsl の GetShadowPositionHClip() を参考に
                float3 positionWS = TransformObjectToWorld(IN.positionOS.xyz);
                float3 normalWS = TransformObjectToWorldNormal(IN.normal);
                float4 positionCS = TransformWorldToHClip(ApplyShadowBias(positionWS, normalWS, _LightDirection));
#if UNITY_REVERSED_Z
                positionCS.z = min(positionCS.z, positionCS.w * UNITY_NEAR_CLIP_VALUE);
#else
                positionCS.z = max(positionCS.z, positionCS.w * UNITY_NEAR_CLIP_VALUE);
#endif
                OUT.positionHCS = positionCS;

                return OUT;
            }

            float4 frag(Varyings IN) : SV_Target {
                return 0.0;
            }

            ENDHLSL
        }
        
        UsePass "Universal Render Pipeline/Lit/DepthOnly"
        UsePass "Universal Render Pipeline/Lit/DepthNormals"
    }
    
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}