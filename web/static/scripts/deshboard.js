$( function() {
  var $winHeight = $( window ).height()
  $( 'indector.container' ).height( $winHeight );
});

$(document).ready(function(){
    $('#test_alexNet,#test_TEST').on('click',function(){
        var _this = this;
        setTimeout(function(){
            if($(_this).parents('div:first').next().css('display')!='block'){
                $(_this).parents('div:first').next().css('display','block');
            }
        },3500);

    });
});